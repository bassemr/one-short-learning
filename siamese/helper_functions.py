import torch
import matplotlib.pyplot as plt # CIFAR-100 normalization stats

MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2023, 0.1994, 0.2010])

def unnormalize(img):
    """Unnormalize a CIFAR-100 image tensor to [0,1]."""
    img = img * STD[:, None, None] + MEAN[:, None, None]
    return torch.clamp(img, 0, 1)

def visualize_pairs(loader, n=5, title="Image Pairs"):
    """
    Visualize the first n pairs from a DataLoader of image pairs.
    Shows img1 and img2 in 2 rows, labels color-coded.
    
    Args:
        loader: DataLoader returning (img1, img2, label)
        n: number of pairs to show
        title: figure title
    """
    img1_batch, img2_batch, labels = next(iter(loader))
    img1_batch, img2_batch, labels = img1_batch[:n], img2_batch[:n], labels[:n]

    # Unnormalize
    img1_batch = torch.stack([unnormalize(img) for img in img1_batch])
    img2_batch = torch.stack([unnormalize(img) for img in img2_batch])

    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # First image
        plt.subplot(2, n, i+1)
        plt.imshow(img1_batch[i].permute(1, 2, 0).numpy())
        color = "green" if labels[i] == 1 else "red"
        plt.title("Same" if labels[i] == 1 else "Diff", color=color, fontsize=10)
        plt.axis("off")

        # Second image
        plt.subplot(2, n, i+1+n)
        plt.imshow(img2_batch[i].permute(1, 2, 0).numpy())
        plt.axis("off")

    plt.suptitle(title)
    plt.show()