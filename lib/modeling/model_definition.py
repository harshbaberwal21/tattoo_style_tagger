"""Module defining the tattoo style tagger model"""

from collections import OrderedDict
import torch
from torch import nn
import torchvision


class ResnetTattooStyleTaggerNN(nn.Module):
    def __init__(self, resnet, out_labels_count):
        super().__init__()
        self.frozen = nn.Sequential(
            OrderedDict([
                (name,layer) for name,layer in resnet.named_children() if name != 'fc'
                ])
                )
        self.unfrozen = nn.Sequential(
            OrderedDict([
                ('fc_resnet', resnet.fc),
                ('fc_resnet_relu', nn.ReLU()),
                ('fc_output', nn.Linear(resnet.fc.out_features, out_labels_count))
                 ])
        )

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


def get_model(out_labels_count):
    """Instantiates ResnetTattooStyleTaggerNN model and freezes all layers
    of the resnet part of the model except the last fully connected layer.

    Returns:
        nn.Module: Instance of the tatto style tagger model.
    """
    resnet_nn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model = ResnetTattooStyleTaggerNN(resnet_nn, out_labels_count)

    for child_layer in model.frozen.children():
        for param in child_layer.parameters():
            param.requires_grad = False

    for child_layer in model.unfrozen.children():
        for param in child_layer.parameters():
            param.requires_grad = True

    return model
