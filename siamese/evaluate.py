import torch
import torch.nn.functional as F

def evaluate_fewshot(model, support_loader, query_loader, device="cpu", batch_size=512):
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
    model.eval()  # ✅ Turn off dropout, batchnorm updates for evaluation
    total_correct, total_samples = 0, 0

    # --- Step 1: Load entire support set (tiny set, so we can take all at once)
    support_images, support_labels = next(iter(support_loader))
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    K = support_images.size(0)  # number of support samples = number of classes

    with torch.no_grad():
        for query_images, query_labels in query_loader:
            query_images, query_labels = query_images.to(device), query_labels.to(device)
            Q = query_images.size(0) # number of query samples in this batch

            # Expand query images and support images so they can be paired
            q_expanded = query_images.unsqueeze(1).expand(-1, K, -1, -1, -1)
            s_expanded = support_images.unsqueeze(0).expand(Q, -1, -1, -1, -1)
            '''
                            | Tensor                        | Shape                   | Meaning                           |
                | ----------------------------- | ----------------------- | --------------------------------- |
                | `query_images`                | `[64, 3, 224, 224]`     | queries                           |
                | `unsqueeze(1)`                | `[64, 1, 3, 224, 224]`  | add a "slot" for K                |
                | `expand(-1, K, …)`            | `[64, 10, 3, 224, 224]` | repeat each query 10 times        |
                | `support_images.unsqueeze(0)` | `[1, 10, 3, 224, 224]`  | add a "slot" for Q                |
                | `expand(Q, -1, …)`            | `[64, 10, 3, 224, 224]` | repeat support set for each query |

            '''
            q_flat = q_expanded.reshape(Q*K, *query_images.shape[1:])  #  q_flat: [640, 3, 224, 224]
            s_flat = s_expanded.reshape(Q*K, *support_images.shape[1:]) # s_flat: [640, 3, 224, 224]
           


            # Process in chunks to avoid OOM
            probs_list = []
            for start in range(0, Q*K, batch_size):
                end = min(start + batch_size, Q*K)
                logits = model(q_flat[start:end], s_flat[start:end])  # [chunk, 1]
                probs = torch.sigmoid(logits).view(-1)
                probs_list.append(probs.detach().cpu())

            # Combine and reshape to [Q, K]
            probs = torch.cat(probs_list).view(Q, K) #probs[i, k] = similarity between query_i and support_k.

            # Pick best support class
            best_idx = torch.argmax(probs, dim=1)   # [Q]
            pred_labels = support_labels[best_idx]

            # Update accuracy
            total_correct += (pred_labels == query_labels).sum().item()
            total_samples += query_labels.size(0)

    acc = total_correct / total_samples
    print(f"[RESULT] Few-shot accuracy: {acc:.4f}")
    return acc



def evaluate_fewshot_distance(model, support_loader, query_loader, device="cuda"):
    """
    Evaluate a SiameseResNet on a few-shot classification task
    using *embedding distances* instead of classifier logits.

    Args:
        model: SiameseResNet (outputs z, dist)
        support_loader: DataLoader containing the support set (1 example per class)
        query_loader: DataLoader containing query samples
        device: "cuda" or "cpu"

    Returns:
        accuracy (float)
    """
    model.eval()
    total_correct, total_samples = 0, 0

    # ---- 1️⃣  Load the entire support set ----
    support_images, support_labels = next(iter(support_loader))
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)

    with torch.no_grad():
        # Compute embeddings for support set
        support_embs = model.forward_once(support_images)   # [K, emb_dim]
        K = support_embs.size(0)

    # ---- 2️⃣  Loop over query batches ----
    with torch.no_grad():
        for query_images, query_labels in query_loader:
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            # Compute query embeddings
            query_embs = model.forward_once(query_images)   # [Q, emb_dim]
            Q = query_embs.size(0)

            # ---- 3️⃣  Compute distances between each query and each support embedding ----
            # Expand for broadcasting
            q_exp = query_embs.unsqueeze(1)    # [Q, 1, D]
            s_exp = support_embs.unsqueeze(0)  # [1, K, D]

            # Euclidean (L2) distance
            dists = torch.norm(q_exp - s_exp, dim=2)   # [Q, K]

            # You could also use cosine similarity if you normalize:
            # dists = 1 - F.cosine_similarity(q_exp, s_exp, dim=2)

            # ---- 4️⃣  Pick nearest support image (smallest distance) ----
            best_idx = torch.argmin(dists, dim=1)       # [Q]
            pred_labels = support_labels[best_idx]      # predicted class for each query

            # ---- 5️⃣  Compute batch accuracy ----
            total_correct += (pred_labels == query_labels).sum().item()
            total_samples += query_labels.size(0)

    # ---- 6️⃣  Final accuracy ----
    acc = total_correct / total_samples
    print(f"[RESULT] Distance-based Few-shot Accuracy: {acc:.4f}")
    return acc
