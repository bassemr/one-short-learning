import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
import copy
import torch.nn.functional as F

class SiameseResNet(nn.Module):
    """
    Siamese network with ResNet18 backbone.
    
    Args:
        embedding_size (int): Size of the embedding vector.
        dropout_p (float): Dropout probability in the embedding head.
    """
    def __init__(self, embedding_size=128, dropout_p=0.3):
        super().__init__()
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        # Keep reference to ResNet layers
        self.resnet = resnet

        # Encoder backbone: remove fc layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Embedding head
        self.fc = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

        # Classifier head (raw logits)
        self.classifier = nn.Linear(embedding_size, 1)

        # Freeze all layers by default
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_until(self, layer_num):
        """
        Unfreeze layers from 'layer_num' onwards.
        layer_num: int, options: 1,2,3,4
        """
        layers_map = {1: self.resnet.layer1,
                      2: self.resnet.layer2,
                      3: self.resnet.layer3,
                      4: self.resnet.layer4}

        # Freeze all layers first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze layers >= layer_num
        for i in range(layer_num, 5):
            for param in layers_map[i].parameters():
                param.requires_grad = True

    def forward_once(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        z = torch.abs(e1 - e2)
        z = self.classifier(z)
        return z

    def summary(self, input_size=(32, 3, 224, 224), verbose=1):
        """
        Wrapper around torchinfo.summary for convenience.

        Args:
            input_size (tuple): Expected input size (batch, channels, height, width).
            verbose (int): 0 = silent, 1 = layer-wise details.
        """
        return summary(self, 
                input_size=[(input_size), (input_size)],  # since we pass two inputs
                verbose=verbose,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])



