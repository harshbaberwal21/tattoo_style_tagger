"""Module defining the tattoo style tagger model"""

from collections import OrderedDict
import torch
from torch import nn
import torchvision


class ResnetTattooStyleTaggerNN(nn.Module):
    """Tattoo style tagger model inheriting from nn.module class"""

    def __init__(self, resnet, out_labels_count):
        super().__init__()
        self.frozen = nn.Sequential(
            OrderedDict(
                [
                    (name, layer)
                    for name, layer in resnet.named_children()
                    if name != "fc"
                ]
            )
        )
        self.unfrozen = nn.Sequential(
            OrderedDict(
                [
                    ("fc_resnet", resnet.fc),
                    ("fc_resnet_relu", nn.ReLU()),
                    ("fc_output", nn.Linear(resnet.fc.out_features, out_labels_count)),
                ]
            )
        )

        nn.init.xavier_normal_(self.unfrozen.fc_resnet.weight)
        nn.init.xavier_normal_(self.unfrozen.fc_output.weight)

    def forward(self, x):
        """Pass forward the data through the layers.

        Args:
            x (torch.tensor): input data tensor

        Returns:
            torch.tensor: Output after passing data through all the layers
        """
        x = self.frozen(x)
        x = torch.flatten(x, 1)
        logits = self.unfrozen(x)
        return logits


def get_model(out_labels_count, pretrained_model_path: str = None):
    """Instantiates ResnetTattooStyleTaggerNN model and freezes all layers
    of the resnet part of the model except the last fully connected layer.

    Returns:
        nn.Module: Instance of the tatto style tagger model.
    """

    if pretrained_model_path:
        model = load_pretrained_model(out_labels_count, pretrained_model_path)
    else:
        resnet_nn = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        model = ResnetTattooStyleTaggerNN(resnet_nn, out_labels_count)

    for child_layer in model.frozen.children():
        for param in child_layer.parameters():
            param.requires_grad = False

    for child_layer in model.unfrozen.children():
        for param in child_layer.parameters():
            param.requires_grad = True

    return model


def load_pretrained_model(out_labels_count, pretrained_model_path):
    """Load the pretrained model."""
    resnet_nn = torchvision.models.resnet18(pretrained=False)
    model = ResnetTattooStyleTaggerNN(resnet_nn, out_labels_count)
    model.load_state_dict(torch.load(pretrained_model_path))
    return model
