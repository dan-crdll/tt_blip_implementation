from model.version_modular.layers.feature_extraction import create_feature_extraction
from model.version_modular.layers.cross_attention_block import create_fusion_layer
from model.version_modular.architecture import Model
from torch import nn 

def create_classifiers():
    print("#### CLASSIFIERS CONFIGURATION ####")

    num_layers_bin = int(input("Number of layers for binary classifier: "))
    num_layers_multi = int(input("Number of layers for multi-label classifier: "))

    hidden_dim_bin = int(input("Hidden dim for binary classifier: "))
    hidden_dim_multi = int(input("Hidden dim for multi-label classifier: "))

    bin_layers = [nn.Linear(768, hidden_dim_bin)]
    for _ in range(num_layers_bin):
        bin_layers.append(nn.Linear(hidden_dim_bin, hidden_dim_bin))
        bin_layers.append(nn.ReLU())
    bin_layers.append(nn.Linear(hidden_dim_bin, 1))

    bin_classifier = nn.Sequential(*bin_layers)

    multi_layers = [nn.Linear(768, hidden_dim_multi)]
    for _ in range(num_layers_multi):
        multi_layers.append(nn.Linear(hidden_dim_multi, hidden_dim_multi))
        multi_layers.append(nn.ReLU())
    multi_layers.append(nn.Linear(hidden_dim_multi, 4))

    multi_classifier = nn.Sequential(*multi_layers)
    return bin_classifier, multi_classifier



def main():
    feature_extraction_layer = create_feature_extraction()
    fusion_layer = create_fusion_layer()
    bin_classifier, multi_classifier = create_classifiers()

    print("##### HYPERPARAMS CONFIGURATION #####")
    lr = int(input("Learning rate: "))

    model = Model(
        feature_extraction_layer,
        fusion_layer,
        bin_classifier,
        multi_classifier,
        lr
    )
