import torch
import torch.nn.functional as F


def masked_infonce_loss(
    q: torch.Tensor,              # (B, N, D) queries from contextual encoder
    k: torch.Tensor,              # (B, N, D) targets (clean)
    mask: torch.Tensor,           # (B, N) bool True=masked positions to score
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE over masked positions:
      loss = -log exp(sim(q_i, k_i)/tau) / sum_j exp(sim(q_i, k_j)/tau)
    where j ranges over all tokens in batch (including itself; that's fine because it's the positive).
    """
    B, N, D = q.shape
    assert k.shape == (B, N, D)
    assert mask.shape == (B, N)

    # Select masked queries
    q_m = q[mask]   # (M, D)
    k_pos = k[mask] # (M, D)
    if q_m.numel() == 0:
        # No masked tokens => loss 0 (avoid NaNs)
        return q.sum() * 0.0

    # Normalize for cosine similarity
    q_m = F.normalize(q_m, dim=-1)
    k_all = F.normalize(k.view(B * N, D), dim=-1)
    k_pos = F.normalize(k_pos, dim=-1)

    # logits: (M, B*N)
    logits = (q_m @ k_all.t()) / temperature

    # labels: index of the positive key in flattened k_all
    # We need the flattened indices of masked positions.
    flat_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)  # (M,)
    labels = flat_idx

    loss = F.cross_entropy(logits, labels)
    return loss
