import torch.nn.functional as F
import torch

def evaluate_all_episodes(model, test_dataset, device="cpu"):
    """
    Evaluate SiameseResNet on episodic test data using vectorized batch computation.
    
    Args:
        model: Trained SiameseResNet.
        test_dataset: Iterable of episodes (dicts with support/query images & labels).
        device: "cuda" or "cpu".
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for episode in test_dataset:
            support_images = episode["support_images"].to(device)   # [K, C, H, W]
            support_labels = episode["support_labels"].to(device)   # [K]
            query_images = episode["query_images"].to(device)       # [Q, C, H, W]
            query_labels = episode["query_labels"].to(device)       # [Q]

            Q, K = query_images.size(0), support_images.size(0)

            # Expand queries and supports into all pairs
            q_expanded = query_images.unsqueeze(1).expand(-1, K, -1, -1, -1)   # [Q, K, C, H, W]
            s_expanded = support_images.unsqueeze(0).expand(Q, -1, -1, -1, -1) # [Q, K, C, H, W]

            # Flatten for batch processing
            q_flat = q_expanded.reshape(Q*K, *query_images.shape[1:])  # [Q*K, C, H, W]
            s_flat = s_expanded.reshape(Q*K, *support_images.shape[1:])# [Q*K, C, H, W]

            # Forward pass through Siamese model
            logits = model(q_flat, s_flat)  # [Q*K, 1]
            probs = torch.sigmoid(logits).view(Q, K)  # reshape back to [Q, K]

            # Pick the support with max similarity for each query
            best_indices = torch.argmax(probs, dim=1)  # [Q]
            pred_labels = support_labels[best_indices] # [Q]

            # Accuracy
            total_correct += (pred_labels == query_labels).sum().item()
            total_samples += query_labels.size(0)

    accuracy = total_correct / total_samples
    print(f"One-shot test accuracy over all episodes: {accuracy:.4f}")
    return accuracy