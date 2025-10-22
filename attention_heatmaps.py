import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import os
from scipy.ndimage import zoom


def rollout_attention(attention_maps: List[torch.Tensor], discard_ratio: float = 0.9) -> torch.Tensor:
    """
    Compute attention rollout across layers.
    
    Args:
        attention_maps: List of attention tensors [BSZ x seq_len x num_patches]
        discard_ratio: Ratio of attention to discard (keep top 1-discard_ratio)
    
    Returns:
        Rolled out attention maps [BSZ x num_patches]
    """
    # Stack attention maps: [num_layers, BSZ, seq_len, num_patches]
    stacked_attention = torch.stack(attention_maps, dim=0)
    
    # Average across sequence length (text tokens)
    # Shape: [num_layers, BSZ, num_patches]
    avg_attention = stacked_attention.mean(dim=2)
    
    # Add identity matrix for residual connections
    num_layers, bsz, num_patches = avg_attention.shape
    eye = torch.eye(num_patches).unsqueeze(0).unsqueeze(0).expand(num_layers, bsz, -1, -1)
    
    # Reshape for matrix operations
    attention_matrices = avg_attention.unsqueeze(-1).expand(-1, -1, -1, num_patches)
    
    # Add residual connections
    attention_matrices = attention_matrices + eye
    
    # Normalize
    attention_matrices = attention_matrices / attention_matrices.sum(dim=-1, keepdim=True)
    
    # Rollout: multiply attention matrices from all layers
    rollout = attention_matrices[0]  # Start with first layer
    for i in range(1, num_layers):
        rollout = torch.matmul(rollout, attention_matrices[i])
    
    # Extract attention for patch tokens (excluding CLS token)
    # Assuming CLS token is at index 0
    patch_attention = rollout[:, 1:, 1:]  # [BSZ, num_patches-1, num_patches-1]
    
    # Sum across the last dimension to get attention for each patch
    final_attention = patch_attention.sum(dim=-1)  # [BSZ, num_patches-1]
    
    return final_attention


def gradient_based_attention(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute gradient-based attention combination.
    
    Args:
        attention_maps: List of attention tensors [BSZ x seq_len x num_patches]
    
    Returns:
        Combined attention maps [BSZ x num_patches]
    """
    # Stack and average across layers and sequence length
    stacked_attention = torch.stack(attention_maps, dim=0)  # [num_layers, BSZ, seq_len, num_patches]
    
    # Average across sequence length (text tokens)
    avg_attention = stacked_attention.mean(dim=2)  # [num_layers, BSZ, num_patches]
    
    # Weighted combination (exponential weighting favoring later layers)
    num_layers = len(attention_maps)
    weights = torch.softmax(torch.arange(num_layers, dtype=torch.float32), dim=0)
    weights = weights.view(-1, 1, 1)  # [num_layers, 1, 1]
    
    # Weighted sum
    combined_attention = (avg_attention * weights).sum(dim=0)  # [BSZ, num_patches]
    
    # Remove CLS token attention (assuming it's at index 0)
    patch_attention = combined_attention[:, 1:]  # [BSZ, num_patches-1]
    
    return patch_attention


def create_heatmap_overlay(image: Image.Image, attention_weights: np.ndarray, 
                          patch_size: int = 16, alpha: float = 0.6, img_size: int = 224) -> Image.Image:
    """
    Create heatmap overlay on the original image.
    
    Args:
        image: PIL Image (any size - will be resized to img_size)
        attention_weights: Attention weights for patches [num_patches]
        patch_size: Size of each patch in pixels
        alpha: Transparency of overlay
        img_size: Target image size (should match ViT input size)
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize image to match the ViT input size
    image_resized = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
    
    # Calculate number of patches per dimension
    patches_per_dim = img_size // patch_size  # Should be 14 for 224x224 with 16x16 patches
    
    # Reshape attention to spatial grid
    if len(attention_weights) != patches_per_dim * patches_per_dim:
        # Handle potential mismatch
        expected_patches = patches_per_dim * patches_per_dim
        if len(attention_weights) > expected_patches:
            attention_weights = attention_weights[:expected_patches]
        else:
            # Pad with zeros if needed
            padding = expected_patches - len(attention_weights)
            attention_weights = np.pad(attention_weights, (0, padding), 'constant')
    
    attention_grid = attention_weights.reshape(patches_per_dim, patches_per_dim)
    
    # Normalize attention weights
    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    
    # Resize attention grid to image size using scipy
    zoom_factor = img_size / patches_per_dim
    heatmap = zoom(attention_grid, zoom_factor, order=1)  # Bilinear interpolation
    
    # Convert to RGB heatmap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Convert PIL image to numpy
    img_array = np.array(image_resized.convert('RGB'))
    
    # Ensure both arrays have the same shape
    if img_array.shape[:2] != heatmap_colored.shape[:2]:
        # Fallback: resize heatmap to match image
        h, w = img_array.shape[:2]
        zoom_factors = (h / heatmap_colored.shape[0], w / heatmap_colored.shape[1], 1)
        heatmap_colored = zoom(heatmap_colored, zoom_factors, order=1).astype(np.uint8)
    
    # Blend images manually
    blended = ((1 - alpha) * img_array + alpha * heatmap_colored).astype(np.uint8)
    
    return Image.fromarray(blended)


def visualize_attention_heatmaps(img: List[Image.Image], 
                               txt: List[str], 
                               z_it: List[torch.Tensor],
                               save_dir: str = "attention_visualizations",
                               method: str = "rollout",
                               alpha: float = 0.6,
                               patch_size: int = 16) -> None:
    """
    Visualize attention heatmaps for a batch of images.
    
    Args:
        img: List of PIL Images (should be 224x224)
        txt: List of corresponding text descriptions
        z_it: List of 6 attention tensors [BSZ x seq_len x num_patches]
        save_dir: Directory to save visualizations
        method: "rollout" or "gradient" for attention combination
        alpha: Transparency of heatmap overlay
        patch_size: Size of patches used in ViT
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = len(img)
    
    # Combine attention maps across layers
    if method == "rollout":
        combined_attention = rollout_attention(z_it)
    elif method == "gradient":
        combined_attention = gradient_based_attention(z_it)
    else:
        raise ValueError("Method must be 'rollout' or 'gradient'")
    
    # Process each image in the batch
    for i in range(batch_size):
        # Get attention weights for this image
        attention_weights = combined_attention[i].detach().cpu().numpy()
        
        # Create heatmap overlay
        heatmap_image = create_heatmap_overlay(
            img[i], attention_weights, patch_size, alpha
        )
        
        # Create visualization with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img[i])
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Pure heatmap
        patches_per_dim = 224 // patch_size
        if len(attention_weights) == patches_per_dim * patches_per_dim:
            heatmap_grid = attention_weights.reshape(patches_per_dim, patches_per_dim)
        else:
            # Handle mismatch
            expected = patches_per_dim * patches_per_dim
            if len(attention_weights) > expected:
                heatmap_grid = attention_weights[:expected].reshape(patches_per_dim, patches_per_dim)
            else:
                padded = np.pad(attention_weights, (0, expected - len(attention_weights)), 'constant')
                heatmap_grid = padded.reshape(patches_per_dim, patches_per_dim)
        
        im = axes[1].imshow(heatmap_grid, cmap='jet', interpolation='bilinear')
        axes[1].set_title("Attention Heatmap")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(heatmap_image)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        # Add text description
        fig.suptitle(f"Text: {txt[i][:100]}{'...' if len(txt[i]) > 100 else ''}", 
                    fontsize=12, y=0.02)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save the visualization
        save_path = os.path.join(save_dir, f"attention_viz_sample_{i:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for sample {i} to {save_path}")


def analyze_attention_statistics(z_it: List[torch.Tensor]) -> dict:
    """
    Analyze attention statistics across layers.
    
    Args:
        z_it: List of 6 attention tensors [BSZ x seq_len x num_patches]
    
    Returns:
        Dictionary with attention statistics
    """
    stats = {}
    
    for layer_idx, attention in enumerate(z_it):
        # Average across batch and sequence length
        avg_attention = attention.mean(dim=(0, 1)).detach().cpu().numpy()
        
        stats[f'layer_{layer_idx}'] = {
            'mean': float(avg_attention.mean()),
            'std': float(avg_attention.std()),
            'max': float(avg_attention.max()),
            'min': float(avg_attention.min()),
            'entropy': float(-np.sum(avg_attention * np.log(avg_attention + 1e-8)))
        }
    
    return stats


def test_model_and_visualize_attention(model, test_dataloader, save_dir="./attention_heatmaps"):
    """
    Test the model and visualize attention maps.
    
    Args:
        model: Trained model with feature extraction and fusion layers
        test_dataloader: DataLoader for test data
        save_dir: Directory to save attention visualizations
    """
    from tqdm.auto import tqdm
    
    model.eval()
    model.to('cpu')
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Process batches and collect attention maps
    batch_count = 0
    max_batches = 5  # Limit visualization to first few batches
    
    for batch in tqdm(test_dataloader):
        if batch_count >= max_batches:
            break
            
        img, txt, (y_bin, y_multi, _), orig = batch
        
        # Move labels to device
        y_bin = y_bin.to(device)
        y_multi = y_multi.to(device)
        
        with torch.no_grad():
            # Forward pass - get predictions and attention maps
            # Returns: (predictions), loss, (representations), (attention_maps)
            (p_bin, p_multi), loss, (z_i, z_t, z), (attn_it_list, attn_tm_list, pv) = model(
                img, txt, orig, y_multi, "Val"
            )
        
        # Visualize attention maps for image branch (attn_it_list)
        # attn_it_list: List of 6 attention tensors [BSZ x seq_len x num_patches]
        visualize_attention_heatmaps(
            img=img,
            txt=txt,
            z_it=attn_it_list,
            save_dir=f"{save_dir}/batch_{batch_count}",
            method="rollout",  # or "gradient"
            alpha=0.6,
            patch_size=16
        )
        
        # Analyze attention statistics
        stats = analyze_attention_statistics(attn_it_list)
        print(f"\nBatch {batch_count} Attention Statistics:")
        for layer, layer_stats in stats.items():
            print(f"  {layer}: mean={layer_stats['mean']:.4f}, "
                  f"std={layer_stats['std']:.4f}, "
                  f"entropy={layer_stats['entropy']:.4f}")
        
        batch_count += 1
    
    print(f"\nAttention visualizations saved to: {save_dir}")


# Example usage with model loading:
from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.architecture import Model
from model.version_modular.utils.load_data import DatasetLoader
import torch
from torch import nn

# Create model components
def create_classifiers(hidden_dim_bin=1024, hidden_dim_multi=1024, 
                      num_layers_bin=3, num_layers_multi=3):
    bin_layers = [nn.Linear(768, hidden_dim_bin)]
    for _ in range(num_layers_bin):
        bin_layers.append(nn.Linear(hidden_dim_bin, hidden_dim_bin))
        bin_layers.append(nn.ReLU())
    bin_layers.append(nn.Linear(hidden_dim_bin, 1))

    multi_layers = [nn.Linear(768, hidden_dim_multi)]
    for _ in range(num_layers_multi):
        multi_layers.append(nn.Linear(hidden_dim_multi, hidden_dim_multi))
        multi_layers.append(nn.ReLU())
    multi_layers.append(nn.Linear(hidden_dim_multi, 4))

    return nn.Sequential(*bin_layers), nn.Sequential(*multi_layers)

# Setup
batch_size = 16
origins = ['washington_post', 'bbc', 'usa_today', 'guardian']
manipulations = ['simswap', 'StyleCLIP', 'infoswap', 'HFGI']

# Load test data
test_dl = DatasetLoader(origins + manipulations, batch_size).get_dataloaders()

# Create model components
feature_extraction_layer = create_feature_extraction()
fusion_layer = create_fusion_layer()
bin_classifier, multi_classifier = create_classifiers()

# Load trained model
model = Model.load_from_checkpoint(
    "Thesis_New/wp46l1i8/checkpoints/epoch=7-step=1632.ckpt",
    feature_extraction_layer=feature_extraction_layer,
    fusion_layer=fusion_layer,
    classifier_bin=bin_classifier,
    classifier_multi=multi_classifier,
    lr=1e-4,
    epoch_tracker=None,
    use_blip=1,
    dataModule=None
)

# Test and visualize
test_model_and_visualize_attention(
    model=model,
    test_dataloader=test_dl,
    save_dir="./attention_heatmaps_new"
)