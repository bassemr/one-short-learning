import torch
import matplotlib.pyplot as plt 
import copy
import torch.nn.functional as F
import torchvision

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





def show_query_support_probabilities(model, test_dataset, device="cpu"):
    """
    Display each query image along with all support images and their similarity probabilities.
    The support with highest similarity is colored green if it matches query label, red if not.
    """
    model.eval()
    with torch.no_grad():
        episode = test_dataset[0]  # single episode
        support_images = episode["support_images"].to(device)   # [K, C, H, W]
        support_labels = episode["support_labels"].to(device)   # [K]
        query_images = episode["query_images"].to(device)       # [Q, C, H, W]
        query_labels = episode["query_labels"].to(device)       # [Q]

        # Compute embeddings
        support_embs = model.forward_once(support_images)       # [K, embedding_dim]
        query_embs = model.forward_once(query_images)           # [Q, embedding_dim]

        # Compute cosine similarity
        sims = F.cosine_similarity(
            query_embs.unsqueeze(1),      # [Q,1,D]
            support_embs.unsqueeze(0),    # [1,K,D]
            dim=2
        )  # [Q, K]

        # Normalize similarity to [0,1] for display
        sims_min, sims_max = sims.min(), sims.max()
        probs = (sims - sims_min) / (sims_max - sims_min)

        # Display each query image
        for i in range(query_images.size(0)):
            q_img = unnormalize(query_images[i].cpu())
            q_label = query_labels[i].item()
            q_probs = probs[i].cpu()

            # Identify the most similar support
            best_idx = torch.argmax(sims[i]).item()
            best_label = support_labels[best_idx].item()
            correct = (best_label == q_label)

            plt.figure(figsize=(12, 3))
            # Query image
            plt.subplot(1, support_images.size(0)+1, 1)
            plt.imshow(torch.permute(q_img, (1,2,0)))
            plt.title(f"Query\nLabel: {q_label}")
            plt.axis('off')

            # Support images with similarity probability
            for j in range(support_images.size(0)):
                s_img = unnormalize(support_images[j].cpu())
                s_label = support_labels[j].item()
                plt.subplot(1, support_images.size(0)+1, j+2)
                plt.imshow(torch.permute(s_img, (1,2,0)))

                # Color title
                if j == best_idx:
                    color = 'green' if correct else 'red'
                else:
                    color = 'black'

                plt.title(f"S{j}\nLabel:{s_label}\nProb:{q_probs[j]:.2f}", color=color)
                plt.axis('off')

            plt.show()



def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()