"""Feature fusion utilities."""
import torch


def fuse_concat(context_emb: torch.Tensor, commonsense_emb: torch.Tensor) -> torch.Tensor:
    """Concatenate context and commonsense embeddings; handles missing commonsense gracefully."""
    if commonsense_emb.numel() == 0:
        return context_emb
    if context_emb.shape[0] != commonsense_emb.shape[0]:
        # Broadcast commonsense if only one commonsense vector is provided
        if commonsense_emb.shape[0] == 1:
            commonsense_emb = commonsense_emb.expand(context_emb.shape[0], -1)
        else:
            raise ValueError("Batch size mismatch between context and commonsense embeddings")
    return torch.cat([context_emb, commonsense_emb], dim=-1)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.norm(x, dim=-1, keepdim=True) + eps
    return x / norm
