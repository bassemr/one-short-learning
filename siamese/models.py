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
        dist = torch.abs(e1 - e2)
        z = self.classifier(dist)
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





class SiameseResNetContrastive(nn.Module):
    """
    Siamese network with ResNet18 backbone for contrastive learning.
    
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

        # Embedding head (two-layer MLP)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, embedding_size)
        )

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
        """
        Compute embedding for a single image batch
        """
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        """
        Compute embeddings for a pair of images
        """
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return e1, e2  # return embeddings for contrastive loss

    def summary(self, input_size=(32, 3, 224, 224), verbose=1):
        """
        Wrapper around torchinfo.summary for convenience.
        """
        return summary(self, 
                input_size=[(input_size), (input_size)],  # two inputs
                verbose=verbose,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])

import torch
import torch.nn as nn

# -------------------------------
# CNN Encoder for 32x32 images
# -------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  # -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x16x16

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), # -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 128x8x8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> 256x4x4
        )

        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)  # -> final embedding
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


# -------------------------------
# Siamese Network 
# -------------------------------
class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = CNNEncoder(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)  # single logit for BCE

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)

        diff = torch.abs(e1 - e2)
        out = self.classifier(diff)  # logits for BCEWithLogitsLoss
        return out
    
    def summary(self, input_size=(32, 3, 32, 32), verbose=0):
        """
        Wrapper around torchinfo.summary for convenience.
        """
        return summary(self, 
                input_size=[(input_size), (input_size)],  # two inputs
                verbose=verbose,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])
