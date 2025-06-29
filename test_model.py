from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.architecture import Model
from torch import nn 
from model.version_modular.utils.load_data import DatasetLoader
from lightning.pytorch.loggers import WandbLogger
import torch
from dgm4_download import download_dgm4
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.classification import BinaryAUROC, MultilabelF1Score, MultilabelAveragePrecision
import yaml
import random
import os
import numpy as np
import lightning as L
from build_difficulty_dataset import create_difficulty_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
import os


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_classifiers(hidden_dim_bin, hidden_dim_multi, num_layers_bin, num_layers_multi):
    print("#### INITIALIZING CLASSIFIERS ####")
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


def main():
    print("##### CONFIGURATION #####")
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)

    pca_50 = PCA(n_components=50)
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=5000)
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=5000)
    tsne_pca = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=5000)

    lr = 1e-4#float(input("Learning rate (e.g., 1e-3): "))
    batch_size = 16#int(input("Batch size: "))
    epochs = 30#int(input("Epochs: "))
    grad_acc = 1#int(input("Gradient accumulation: "))
    gpus_input = "0,1"#input("GPUs (comma-separated, no spaces): ")
    grad_clip = 1#float(input("Gradient clipping: "))
    curriculum = 1#int(input("Use curriculum learning: Y (1) | N (0): "))
    num_layers_bin = 3#int(input("Number of layers for binary classifier: "))
    num_layers_multi = 3#int(input("Number of layers for multi-label classifier: "))
    hidden_dim_bin = 1024#int(input("Hidden dim for binary classifier: "))
    hidden_dim_multi = 1024#int(input("Hidden dim for multi-label classifier: "))
    blip = 1#int(input("Use Blip: Y (1) | N(0): "))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_input
    gpus = [int(gpu) for gpu in gpus_input.split(",")]

    seed_everything()

    origins = ['washington_post', 'bbc', 'usa_today', 'guardian']
    manipulations = ['simswap', 'StyleCLIP', 'infoswap', 'HFGI']

    test_dl = DatasetLoader(origins + manipulations, batch_size).get_dataloaders()
    et = None
    datamodule = None

    feature_extraction_layer = create_feature_extraction()
    fusion_layer = create_fusion_layer()
    bin_classifier, multi_classifier = create_classifiers(
        hidden_dim_bin, hidden_dim_multi, num_layers_bin, num_layers_multi
    )

    torch.set_float32_matmul_precision('high')

    model = Model.load_from_checkpoint("Thesis_New/wp46l1i8/checkpoints/epoch=7-step=1632.ckpt",
        feature_extraction_layer=feature_extraction_layer,
        fusion_layer=fusion_layer,
        classifier_bin=bin_classifier,
        classifier_multi=multi_classifier,
        lr=lr,
        epoch_tracker=et, 
        use_blip=blip,
        dataModule=datamodule
    )
    model.eval()

    # Determine device
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # Move all metrics to the same device as the model
    acc_bin = Accuracy(task='binary').to(device)
    f1_bin = F1Score(task='binary').to(device)
    auc_bin = BinaryAUROC().to(device)

    # Multilabel classification metrics (4 labels)
    acc_multi = Accuracy(task='multilabel', num_labels=4).to(device)
    cf1_multi = MultilabelF1Score(num_labels=4, average='macro').to(device)
    of1_multi = MultilabelF1Score(num_labels=4, average='micro').to(device)
    map_multi = MultilabelAveragePrecision(num_labels=4).to(device)

    confusion_matrix = torch.zeros((2, 2)).long().to(device)
    representations = []
    all_representations = []  # Collect all representations for PCA

    for batch in tqdm(test_dl):
        img, txt, (y_bin, y_multi, _), orig = batch

        
        # Move only the label tensors to device (img and txt are handled by the model)
        y_bin = y_bin.to(device)
        y_multi = y_multi.to(device)
        
        # Convert binary labels to integers for metrics that expect integer targets
        y_bin_int = (y_bin > 0.5).long()  # Convert float labels to 0/1 integers

        with torch.no_grad():
            (p_bin, p_multi), loss, (z_i, z_t, z), (attn_it_list, attn_tm_list, pv) = model(img, txt, orig, y_multi, "Val")

        pred_bin_sigmoid = torch.sigmoid(p_bin)
        pred_multi_sigmoid = torch.sigmoid(p_multi)


        # visualize_text_attention_heatmaps(
        #     img=img,
        #     txt=txt,
        #     z_tm=attn_tm_list,
        #     z_it=attn_it_list,
        #     save_dir="./text_attention_heatmaps",
        #     method="gradient",  # or "gradient"
        #     tokenizer=model.feature_extraction.feature_extractor_txt.text_encoder.tokenizer  # or your tokenizer
        # )


        # Fix confusion matrix calculation
        pred_bin_labels = (pred_bin_sigmoid > 0.5).long()
        true_bin_labels = y_bin_int  # Use the integer version
        
        for pred, true in zip(pred_bin_labels, true_bin_labels):
            confusion_matrix[true, pred] += 1

        # Collect representations for PCA (move to CPU)
        all_representations.append(z.cpu().numpy())
        representations.append({
            'bin_cls': y_bin.cpu().numpy(), 
            'mult_cls': y_multi.cpu().numpy()
        })

        # Update metrics (using integer labels for binary classification)
        acc_bin.update(pred_bin_sigmoid, y_bin_int)
        f1_bin.update(pred_bin_sigmoid, y_bin_int)
        auc_bin.update(pred_bin_sigmoid, y_bin_int)  # AUC also needs integer labels

        acc_multi.update(pred_multi_sigmoid, y_multi)
        cf1_multi.update(pred_multi_sigmoid, y_multi)
        of1_multi.update(pred_multi_sigmoid, y_multi)
        map_multi.update(pred_multi_sigmoid, y_multi.long())
    
    # Fit PCA and t-SNE on all representations
    all_representations = np.vstack(all_representations)
    pca_2d_result = pca_2d.fit_transform(all_representations)
    pca_3d_result = pca_3d.fit_transform(all_representations)
    
    print("Computing 2D t-SNE... (this may take a while)")
    tsne_2d_result = tsne_2d.fit_transform(all_representations)
    
    print("Computing 3D t-SNE... (this may take a while)")
    tsne_3d_result = tsne_3d.fit_transform(all_representations)

    print("Computing 2D t-SNE from PCA Reduction... (this may take a while)")
    pca_res = pca_50.fit_transform(all_representations)
    tsne_pca_result = tsne_pca.fit_transform(pca_res)    

    # Add PCA and t-SNE results to representations
    start_idx = 0
    for i, rep in enumerate(representations):
        batch_size = rep['bin_cls'].shape[-1]
        rep['point_2d'] = pca_2d_result[start_idx:start_idx + batch_size]
        rep['point_3d'] = pca_3d_result[start_idx:start_idx + batch_size]
        rep['tsne_2d_point'] = tsne_2d_result[start_idx:start_idx + batch_size]
        rep['tsne_3d_point'] = tsne_3d_result[start_idx:start_idx + batch_size]
        rep['tsne_pca_point'] = tsne_pca_result[start_idx:start_idx + batch_size]
        start_idx += batch_size
    
    with open("test_results.txt", "w") as f:
        f.write(f"Binary Accuracy: {acc_bin.compute().item():.4f}\n")
        f.write(f"Binary F1 Score: {f1_bin.compute().item():.4f}\n")
        f.write(f"Binary AUROC: {auc_bin.compute().item():.4f}\n")
        f.write(f"Multilabel Accuracy: {acc_multi.compute().item():.4f}\n")
        f.write(f"Multilabel Macro F1: {cf1_multi.compute().item():.4f}\n")
        f.write(f"Multilabel Micro F1: {of1_multi.compute().item():.4f}\n")
        f.write(f"Multilabel mAP: {map_multi.compute().item():.4f}\n")

    # Normalize the confusion matrix (move to CPU for plotting)
    conf_matrix = confusion_matrix.float().cpu()
    conf_matrix_norm = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)
    conf_matrix_norm = conf_matrix_norm.numpy()

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Extract data for visualization
    all_points_2d = []
    all_points_3d = []
    all_tsne_2d_points = []
    all_tsne_pca_points = []
    all_tsne_3d_points = []
    all_bin_labels = []
    all_multi_labels = []

    for rep in representations:
        all_points_2d.append(rep['point_2d'])
        all_points_3d.append(rep['point_3d'])
        all_tsne_2d_points.append(rep['tsne_2d_point'])
        all_tsne_3d_points.append(rep['tsne_3d_point'])
        all_tsne_pca_points.append(rep['tsne_pca_point'])
        all_bin_labels.append(rep['bin_cls'])
        all_multi_labels.append(rep['mult_cls'])

    # Convert to numpy arrays
    all_points_2d = np.vstack(all_points_2d)
    all_points_3d = np.vstack(all_points_3d)
    all_tsne_2d_points = np.vstack(all_tsne_2d_points)
    all_tsne_pca_points = np.vstack(all_tsne_pca_points)
    all_tsne_3d_points = np.vstack(all_tsne_3d_points)
    all_bin_labels = np.concatenate(all_bin_labels)
    all_multi_labels = np.vstack(all_multi_labels)

    # 3D PCA Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(all_points_3d[:, 0], all_points_3d[:, 1], all_points_3d[:, 2],
                        c=all_bin_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Binary Class')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    plt.tight_layout()
    plt.savefig('pca_3d_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3D PCA Plot with multilabel classification for fake samples (y_bin == 1)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate real and fake samples
    real_mask = all_bin_labels == 0
    fake_mask = all_bin_labels == 1
    
    # Plot real samples in one color
    ax.scatter(all_points_3d[real_mask, 0], all_points_3d[real_mask, 1], all_points_3d[real_mask, 2],
               c='blue', alpha=0.7, label='Real', s=50)
    
    # For fake samples, create unique identifiers based on multi-label combinations
    fake_points_3d = all_points_3d[fake_mask]
    fake_multi_labels = all_multi_labels[fake_mask]
    
    # Convert multi-label arrays to unique identifiers
    fake_label_strings = [''.join(map(str, row.astype(int))) for row in fake_multi_labels]
    unique_patterns = list(set(fake_label_strings))
    
    # Create a color map for different multi-label patterns
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_patterns)))
    
    for i, pattern in enumerate(unique_patterns):
        pattern_mask = np.array(fake_label_strings) == pattern
        pattern_points = fake_points_3d[pattern_mask]
        
        if len(pattern_points) > 0:
            ax.scatter(pattern_points[:, 0], pattern_points[:, 1], pattern_points[:, 2],
                       c=[colors[i]], alpha=0.7, 
                       label=f'Fake Pattern: {pattern}', s=50)
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('pca_3d_multilabel_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2D PCA Plot with multilabel classification for fake samples (y_bin == 1)
    plt.figure(figsize=(10, 8))
    
    # Separate real and fake samples
    real_mask = all_bin_labels == 0
    fake_mask = all_bin_labels == 1
    
    # Plot real samples in one color
    plt.scatter(all_points_2d[real_mask, 0], all_points_2d[real_mask, 1], 
               c='blue', alpha=0.7, label='Real', s=50)
    
    # For fake samples, create unique identifiers based on multi-label combinations
    fake_points = all_points_2d[fake_mask]
    fake_multi_labels = all_multi_labels[fake_mask]
    
    # Convert multi-label arrays to unique identifiers
    # Each row becomes a string representation of the binary pattern
    fake_label_strings = [''.join(map(str, row.astype(int))) for row in fake_multi_labels]
    unique_patterns = list(set(fake_label_strings))
    
    # Create a color map for different multi-label patterns
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_patterns)))
    
    for i, pattern in enumerate(unique_patterns):
        pattern_mask = np.array(fake_label_strings) == pattern
        pattern_points = fake_points[pattern_mask]
        
        if len(pattern_points) > 0:
            plt.scatter(pattern_points[:, 0], pattern_points[:, 1], 
                       c=[colors[i]], alpha=0.7, 
                       label=f'Fake Pattern: {pattern}', s=50)
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_multilabel_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # t-SNE visualization (binary classification)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_tsne_2d_points[:, 0], all_tsne_2d_points[:, 1], 
                        c=all_bin_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Binary Class')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_2d_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # t-SNE visualization (pca)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_tsne_pca_points[:, 0], all_tsne_pca_points[:, 1], 
                        c=all_bin_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Binary Class')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_pca_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3D t-SNE Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(all_tsne_3d_points[:, 0], all_tsne_3d_points[:, 1], all_tsne_3d_points[:, 2],
                        c=all_bin_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Binary Class')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.tight_layout()
    plt.savefig('tsne_3d_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # t-SNE visualization with multilabel classification for fake samples
    plt.figure(figsize=(10, 8))
    
    # Separate real and fake samples
    real_mask = all_bin_labels == 0
    fake_mask = all_bin_labels == 1
    
    # Plot real samples in one color
    plt.scatter(all_tsne_2d_points[real_mask, 0], all_tsne_2d_points[real_mask, 1], 
               c='blue', alpha=0.7, label='Real', s=50)
    
    # For fake samples, create unique identifiers based on multi-label combinations
    fake_points = all_tsne_2d_points[fake_mask]
    fake_multi_labels = all_multi_labels[fake_mask]
    
    # Convert multi-label arrays to unique identifiers
    fake_label_strings = [''.join(map(str, row.astype(int))) for row in fake_multi_labels]
    unique_patterns = list(set(fake_label_strings))
    
    # Create a color map for different multi-label patterns
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_patterns)))
    
    for i, pattern in enumerate(unique_patterns):
        pattern_mask = np.array(fake_label_strings) == pattern
        pattern_points = fake_points[pattern_mask]
        
        if len(pattern_points) > 0:
            plt.scatter(pattern_points[:, 0], pattern_points[:, 1], 
                       c=[colors[i]], alpha=0.7, 
                       label=f'Fake Pattern: {pattern}', s=50)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_multilabel_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # t-SNE visualization with multilabel classification for fake samples
    plt.figure(figsize=(10, 8))
    
    # Separate real and fake samples
    real_mask = all_bin_labels == 0
    fake_mask = all_bin_labels == 1
    
    # Plot real samples in one color
    plt.scatter(all_tsne_pca_points[real_mask, 0], all_tsne_pca_points[real_mask, 1], 
               c='blue', alpha=0.7, label='Real', s=50)
    
    # For fake samples, create unique identifiers based on multi-label combinations
    fake_points = all_tsne_pca_points[fake_mask]
    fake_multi_labels = all_multi_labels[fake_mask]
    
    # Convert multi-label arrays to unique identifiers
    fake_label_strings = [''.join(map(str, row.astype(int))) for row in fake_multi_labels]
    unique_patterns = list(set(fake_label_strings))
    
    # Create a color map for different multi-label patterns
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_patterns)))
    
    for i, pattern in enumerate(unique_patterns):
        pattern_mask = np.array(fake_label_strings) == pattern
        pattern_points = fake_points[pattern_mask]
        
        if len(pattern_points) > 0:
            plt.scatter(pattern_points[:, 0], pattern_points[:, 1], 
                       c=[colors[i]], alpha=0.7, 
                       label=f'Fake Pattern: {pattern}', s=50)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_multilabel_pca_representations.png', dpi=300, bbox_inches='tight')
    plt.close()




import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
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
    
    # Weighted combination (you can learn these weights or use heuristics)
    # Here we use exponential weighting favoring later layers
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
    
    # Blend images manually (replacing cv2.addWeighted)
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

# Example usage:
"""
# Assuming you have your data loaded:
# img: List of PIL Images
# txt: List of strings  
# z_it: List of 6 torch.Tensors with shape [BSZ x seq_len x num_patches]

visualize_attention_heatmaps(
    img=img,
    txt=txt, 
    z_it=z_it,
    save_dir="./attention_heatmaps",
    method="rollout",  # or "gradient"
    alpha=0.6,
    patch_size=16
)

# Analyze attention statistics
stats = analyze_attention_statistics(z_it)
print("Attention Statistics:", stats)
"""









import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import os
from scipy.ndimage import zoom
import seaborn as sns

def rollout_attention(attention_maps: List[torch.Tensor], discard_ratio: float = 0.9) -> torch.Tensor:
    """
    Compute attention rollout across layers for text tokens.
    
    Args:
        attention_maps: List of attention tensors [BSZ x seq_len x num_patches/num_multimodal_dim]
        discard_ratio: Ratio of attention to discard (keep top 1-discard_ratio)
    
    Returns:
        Rolled out attention maps [BSZ x seq_len]
    """
    # Stack attention maps: [num_layers, BSZ, seq_len, feature_dim]
    stacked_attention = torch.stack(attention_maps, dim=0)
    
    # Average across feature dimension (patches or multimodal dimensions)
    # Shape: [num_layers, BSZ, seq_len]
    avg_attention = stacked_attention.mean(dim=-1)
    
    # Weighted combination favoring later layers
    num_layers = len(attention_maps)
    weights = torch.softmax(torch.arange(num_layers, dtype=torch.float32) * 2, dim=0)
    weights = weights.view(-1, 1, 1)  # [num_layers, 1, 1]
    
    # Weighted sum across layers
    combined_attention = (avg_attention * weights).sum(dim=0)  # [BSZ, seq_len]
    
    return combined_attention

def gradient_based_attention(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute gradient-based attention combination for text tokens.
    
    Args:
        attention_maps: List of attention tensors [BSZ x seq_len x feature_dim]
    
    Returns:
        Combined attention maps [BSZ x seq_len]
    """
    # Stack and average across feature dimension
    stacked_attention = torch.stack(attention_maps, dim=0)  # [num_layers, BSZ, seq_len, feature_dim]
    
    # Average across feature dimension
    avg_attention = stacked_attention.mean(dim=-1)  # [num_layers, BSZ, seq_len]
    
    # Exponential weighting favoring later layers
    num_layers = len(attention_maps)
    weights = torch.softmax(torch.arange(num_layers, dtype=torch.float32) * 1.5, dim=0)
    weights = weights.view(-1, 1, 1)  # [num_layers, 1, 1]
    
    # Weighted sum
    combined_attention = (avg_attention * weights).sum(dim=0)  # [BSZ, seq_len]
    
    return combined_attention

def create_text_heatmap_image(tokens: List[str], 
                            attention_scores: np.ndarray,
                            title: str = "Text Attention",
                            width: int = 1200,
                            height: int = 300,
                            font_size: int = 24) -> Image.Image:
    """
    Create a text heatmap as an image with tokens colored by attention intensity.
    
    Args:
        tokens: List of text tokens
        attention_scores: Attention scores for each token [seq_len]
        title: Title for the visualization
        width: Image width 
        height: Image height
        font_size: Font size for tokens
    
    Returns:
        PIL Image with text heatmap
    """
    # Normalize attention scores
    attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
    
    # Create image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw title
    title_font_size = int(font_size * 1.2)
    try:
        title_font = ImageFont.truetype("arial.ttf", title_font_size)
    except:
        title_font = font
    
    draw.text((10, 10), title, fill='black', font=title_font)
    
    # Calculate text layout
    y_start = 60
    x_margin = 20
    line_height = font_size + 10
    
    # Get colormap
    colormap = plt.cm.get_cmap('Reds')
    
    # Draw tokens with background colors based on attention
    x = x_margin
    y = y_start
    max_width = width - 2 * x_margin
    
    for i, (token, score) in enumerate(zip(tokens, attention_scores)):
        # Get color based on attention score
        color_rgba = colormap(score)
        color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
        
        # Measure token text
        try:
            bbox = draw.textbbox((0, 0), token, font=font)
            token_width = bbox[2] - bbox[0]
            token_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            token_width = len(token) * (font_size // 2)
            token_height = font_size
        
        # Check if we need to wrap to next line
        if x + token_width > max_width:
            x = x_margin
            y += line_height
        
        # Ensure we don't exceed image height
        if y + token_height > height - 20:
            break
        
        # Draw background rectangle
        padding = 4
        rect_coords = [
            x - padding, 
            y - padding,
            x + token_width + padding,
            y + token_height + padding
        ]
        draw.rectangle(rect_coords, fill=color_rgb, outline='gray')
        
        # Draw token text (black for light backgrounds, white for dark)
        text_color = 'white' if score > 0.6 else 'black'
        draw.text((x, y), token, fill=text_color, font=font)
        
        # Move to next position
        x += token_width + 15
    
    return img

def create_text_heatmap_plot(tokens: List[str], 
                           attention_scores: np.ndarray,
                           title: str = "Text Attention",
                           figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """
    Create a matplotlib heatmap for text tokens.
    
    Args:
        tokens: List of text tokens
        attention_scores: Attention scores for each token [seq_len]
        title: Title for the plot
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize attention scores
    attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
    
    # Create heatmap matrix
    attention_matrix = attention_scores.reshape(1, -1)
    
    # Create heatmap
    im = ax.imshow(attention_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Score', rotation=270, labelpad=15)
    
    # Add value annotations
    for i, score in enumerate(attention_scores):
        color = 'white' if score > 0.5 else 'black'
        ax.text(i, 0, f'{score:.3f}', ha='center', va='center', 
               color=color, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and punctuation."""
    import re
    # Split on whitespace and common punctuation, but keep the punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def visualize_text_attention_heatmaps(img: List[Image.Image], 
                                    txt: List[str], 
                                    z_tm: List[torch.Tensor],
                                    z_it: List[torch.Tensor],
                                    save_dir: str = "text_attention_visualizations",
                                    method: str = "rollout",
                                    tokenizer=None) -> None:
    """
    Visualize text attention heatmaps for both multimodal and image branches.
    
    Args:
        img: List of PIL Images
        txt: List of corresponding text descriptions
        z_tm: List of 6 attention tensors for text-multimodal [BSZ x seq_len x num_multimodal_dim]
        z_it: List of 6 attention tensors for text-image [BSZ x seq_len x num_patches]
        save_dir: Directory to save visualizations
        method: "rollout" or "gradient" for attention combination
        tokenizer: Optional tokenizer for getting tokens
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = len(img)
    
    # Combine attention maps across layers for both branches
    if method == "rollout":
        tm_attention = rollout_attention(z_tm)  # [BSZ x seq_len]
        ti_attention = rollout_attention(z_it)  # [BSZ x seq_len]
    elif method == "gradient":
        tm_attention = gradient_based_attention(z_tm)
        ti_attention = gradient_based_attention(z_it)
    else:
        raise ValueError("Method must be 'rollout' or 'gradient'")
    
    # Process each sample in the batch
    for i in range(batch_size):
        print(f"Processing sample {i+1}/{batch_size}")
        
        # Get text and tokenize
        text = txt[i]
        if tokenizer is not None:
            tokens = tokenizer.tokenize(text)
        else:
            tokens = simple_tokenize(text)
        
        # Get attention scores for this sample
        tm_scores = tm_attention[i].detach().cpu().numpy()
        ti_scores = ti_attention[i].detach().cpu().numpy()
        
        # Ensure we have the right number of tokens
        min_len = min(len(tokens), len(tm_scores), len(ti_scores))
        tokens = tokens[:min_len]
        tm_scores = tm_scores[:min_len]
        ti_scores = ti_scores[:min_len]
        
        # Create visualizations
        
        # 1. Text-Multimodal branch heatmap (as image)
        tm_heatmap_img = create_text_heatmap_image(
            tokens, tm_scores, 
            title=f"Text-Multimodal Attention (Sample {i+1})",
            width=1400, height=200
        )
        
        # 2. Text-Image branch heatmap (as image)
        ti_heatmap_img = create_text_heatmap_image(
            tokens, ti_scores,
            title=f"Text-Image Attention (Sample {i+1})",
            width=1400, height=200
        )
        
        # 3. Create comparison plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Show original image
        axes[0].imshow(img[i])
        axes[0].set_title(f"Original Image - Sample {i+1}", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Text-Multimodal heatmap
        tm_matrix = tm_scores.reshape(1, -1)
        tm_norm = (tm_scores - tm_scores.min()) / (tm_scores.max() - tm_scores.min() + 1e-8)
        
        im1 = axes[1].imshow(tm_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=tm_scores.max())
        axes[1].set_xticks(range(len(tokens)))
        axes[1].set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        axes[1].set_yticks([])
        axes[1].set_title('Text-Multimodal Attention', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for j, score in enumerate(tm_norm):
            color = 'white' if score > 0.5 else 'black'
            axes[1].text(j, 0, f'{tm_scores[j]:.3f}', ha='center', va='center', 
                        color=color, fontsize=8, fontweight='bold')
        
        # Text-Image heatmap
        ti_matrix = ti_scores.reshape(1, -1)
        ti_norm = (ti_scores - ti_scores.min()) / (ti_scores.max() - ti_scores.min() + 1e-8)
        
        im2 = axes[2].imshow(ti_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=ti_scores.max())
        axes[2].set_xticks(range(len(tokens)))
        axes[2].set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        axes[2].set_yticks([])
        axes[2].set_title('Text-Image Attention', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for j, score in enumerate(ti_norm):
            color = 'white' if score > 0.5 else 'black'
            axes[2].text(j, 0, f'{ti_scores[j]:.3f}', ha='center', va='center', 
                        color=color, fontsize=8, fontweight='bold')
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[1], shrink=0.8, label='Attention Score')
        plt.colorbar(im2, ax=axes[2], shrink=0.8, label='Attention Score')
        
        # Add text description
        fig.suptitle(f"Text: {text[:150]}{'...' if len(text) > 150 else ''}", 
                    fontsize=12, y=0.02)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save visualizations
        comparison_path = os.path.join(save_dir, f"text_attention_comparison_sample_{i:03d}.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual heatmap images
        tm_path = os.path.join(save_dir, f"text_multimodal_heatmap_sample_{i:03d}.png")
        tm_heatmap_img.save(tm_path)
        
        ti_path = os.path.join(save_dir, f"text_image_heatmap_sample_{i:03d}.png")
        ti_heatmap_img.save(ti_path)
        
        print(f"Saved visualizations for sample {i}:")
        print(f"  - Comparison: {comparison_path}")
        print(f"  - Text-Multimodal: {tm_path}")
        print(f"  - Text-Image: {ti_path}")

def analyze_text_attention_statistics(z_tm: List[torch.Tensor], 
                                    z_it: List[torch.Tensor],
                                    txt: List[str],
                                    tokenizer=None) -> dict:
    """
    Analyze text attention statistics across layers and samples.
    
    Args:
        z_tm: List of 6 attention tensors for text-multimodal
        z_it: List of 6 attention tensors for text-image  
        txt: List of text descriptions
        tokenizer: Optional tokenizer
    
    Returns:
        Dictionary with attention statistics
    """
    stats = {'text_multimodal': {}, 'text_image': {}}
    
    # Analyze each branch
    for branch_name, attention_maps in [('text_multimodal', z_tm), ('text_image', z_it)]:
        for layer_idx, attention in enumerate(attention_maps):
            # Average across feature dimension, then across batch
            layer_attention = attention.mean(dim=-1)  # [BSZ x seq_len]
            avg_attention = layer_attention.mean(dim=0).detach().cpu().numpy()  # [seq_len]
            
            stats[branch_name][f'layer_{layer_idx}'] = {
                'mean': float(avg_attention.mean()),
                'std': float(avg_attention.std()),
                'max': float(avg_attention.max()),
                'min': float(avg_attention.min()),
                'entropy': float(-np.sum(avg_attention * np.log(avg_attention + 1e-8)))
            }
    
    # Token-level statistics
    stats['token_analysis'] = []
    
    # Combine attention across layers
    tm_combined = rollout_attention(z_tm)
    ti_combined = rollout_attention(z_it)
    
    for i, text in enumerate(txt):
        if tokenizer is not None:
            tokens = tokenizer.tokenize(text)
        else:
            tokens = simple_tokenize(text)
        
        tm_scores = tm_combined[i].detach().cpu().numpy()
        ti_scores = ti_combined[i].detach().cpu().numpy()
        
        min_len = min(len(tokens), len(tm_scores), len(ti_scores))
        
        sample_stats = {
            'sample_idx': i,
            'num_tokens': min_len,
            'tm_max_token': tokens[np.argmax(tm_scores[:min_len])],
            'tm_max_score': float(tm_scores[:min_len].max()),
            'ti_max_token': tokens[np.argmax(ti_scores[:min_len])],
            'ti_max_score': float(ti_scores[:min_len].max()),
            'correlation': float(np.corrcoef(tm_scores[:min_len], ti_scores[:min_len])[0,1])
        }
        stats['token_analysis'].append(sample_stats)
    
    return stats

# Example usage:
"""
# Assuming you have your data loaded:
# img: List of PIL Images
# txt: List of strings  
# z_tm: List of 6 torch.Tensors with shape [BSZ x seq_len x num_multimodal_dim]
# z_it: List of 6 torch.Tensors with shape [BSZ x seq_len x num_patches]

visualize_text_attention_heatmaps(
    img=img,
    txt=txt,
    z_tm=z_tm,
    z_it=z_it,
    save_dir="./text_attention_heatmaps",
    method="rollout",  # or "gradient"
    tokenizer=None  # or your tokenizer
)

# Analyze attention statistics
stats = analyze_text_attention_statistics(z_tm, z_it, txt)
print("Text Attention Statistics:", stats)
"""











if __name__ == "__main__":
    main()