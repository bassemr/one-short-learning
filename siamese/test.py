import torch
import torch.nn.functional as F

def evaluate_fewshot(model, support_loader, query_loader, device="cuda", batch_size=512):
    """
    Evaluate a Siamese network on a 1-shot N-way task.

    Args:
        model: SiameseResNet (outputs logits)
        support_loader: DataLoader with support set (1 per class)
        query_loader: DataLoader with queries
        device: "cuda" or "cpu"
        batch_size: chunk size for processing query-support pairs

    Returns:
        accuracy (float)
    """
    model.eval()
    total_correct, total_samples = 0, 0

    # Load support set (tiny, so just take all)
    support_images, support_labels = next(iter(support_loader))
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    K = support_images.size(0)  # number of classes

    with torch.no_grad():
        for query_images, query_labels in query_loader:
            query_images, query_labels = query_images.to(device), query_labels.to(device)
            Q = query_images.size(0)

            # Expand into Q*K pairs
            q_expanded = query_images.unsqueeze(1).expand(-1, K, -1, -1, -1)
            s_expanded = support_images.unsqueeze(0).expand(Q, -1, -1, -1, -1)

            q_flat = q_expanded.reshape(Q*K, *query_images.shape[1:])
            s_flat = s_expanded.reshape(Q*K, *support_images.shape[1:])

            # Process in chunks to avoid OOM
            probs_list = []
            for start in range(0, Q*K, batch_size):
                end = min(start + batch_size, Q*K)
                logits = model(q_flat[start:end], s_flat[start:end])  # [chunk, 1]
                probs = torch.sigmoid(logits).view(-1)
                probs_list.append(probs.detach().cpu())

            # Combine and reshape to [Q, K]
            probs = torch.cat(probs_list).view(Q, K)

            # Pick best support class
            best_idx = torch.argmax(probs, dim=1)   # [Q]
            pred_labels = support_labels[best_idx]

            # Update accuracy
            total_correct += (pred_labels == query_labels).sum().item()
            total_samples += query_labels.size(0)

    acc = total_correct / total_samples
    print(f"[RESULT] Few-shot accuracy: {acc:.4f}")
    return acc
