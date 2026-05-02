from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


POSITIVE_KIND = 0
NEGATIVE_KIND = 1


# =========================
# NEW: Weighted BCE (stable, logits version)
# =========================
def weighted_bce_with_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """
    Numerically stable weighted BCE with logits.

    pos_weight: weight for positive pixels
    neg_weight: weight for negative pixels (KEY for reducing FP)
    """
    # Stable BCE with logits
    max_val = (-logits).clamp(min=0)
    loss = logits - logits * target + max_val + \
        torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val))

    # Apply weights
    weight = pos_weight * target + neg_weight * (1 - target)
    loss = weight * loss

    return loss.mean()


def _zero_scalar_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=x.device, dtype=x.dtype)


def _select_bt(tensor_bt: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    if mask_bt.any():
        return tensor_bt[mask_bt]
    return tensor_bt[:0]


def meta_only_loss(
    meta_logits: torch.Tensor,
    meta_t: torch.Tensor,
    phase: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
    apply_sigmoid_to_meta: bool = False,
    sample_kind: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-head meta-only loss.

    Positive samples (sample_kind==0):
      - meta loss only on phase==0 frames

    Negative samples (sample_kind==1):
      - meta loss on all frames against zero target

    NOTE:
      - Now uses weighted BCE → neg_weight is ACTIVE (fixes FP explosion)
    """
    del apply_sigmoid_to_meta

    if meta_logits.ndim != 5:
        raise ValueError("Expected logits to have shape [B,T,1,H,W].")
    if meta_t.shape != meta_logits.shape:
        raise ValueError("Target shapes must match logits shapes.")
    if phase.ndim != 2:
        raise ValueError("Expected phase to have shape [B,T].")

    B, T = phase.shape

    meta_logits_bt = meta_logits.reshape(B * T, *meta_logits.shape[2:])
    meta_t_bt = meta_t.reshape(B * T, *meta_t.shape[2:])
    phase_bt = phase.reshape(-1)

    if sample_kind is None:
        meta_idx = (phase_bt == 0)
    else:
        if sample_kind.ndim == 0:
            sample_kind = sample_kind.unsqueeze(0)
        if sample_kind.ndim != 1 or sample_kind.shape[0] != B:
            raise ValueError("sample_kind must have shape [B].")

        sample_kind_bt = sample_kind[:, None].expand(B, T).reshape(-1)

        positive_bt = (sample_kind_bt == POSITIVE_KIND)
        negative_bt = (sample_kind_bt == NEGATIVE_KIND)

        meta_idx = (positive_bt & (phase_bt == 0)) | negative_bt

    if meta_idx.any():
        meta_sel_logits = _select_bt(meta_logits_bt, meta_idx)
        meta_sel_t = _select_bt(meta_t_bt, meta_idx)

        # core changed：using weighted BCE
        meta_loss = weighted_bce_with_logits_loss(
            meta_sel_logits,
            meta_sel_t,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
        )
    else:
        meta_loss = _zero_scalar_like(meta_logits)

    return meta_loss, meta_loss.detach()
