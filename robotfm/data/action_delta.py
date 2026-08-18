"""Joint-increment action targets: predict Δq = action − q_now, grippers absolute.

``q_now`` is the current observation state (last obs frame). Enabled by
``policy.predict_joint_delta``. Action normalization then uses residual stats
(overwritten onto ``action_mean`` / ``action_std`` / min / max).
"""

from __future__ import annotations

import numpy as np
import torch

from robotfm.data.stats import denormalize, normalize


def joint_mask_from_names(action_names: list[str] | None, action_dim: int) -> np.ndarray:
    """True on joint dims, False on gripper dims."""
    names = list(action_names or [])
    if len(names) == int(action_dim):
        grip = np.array(["gripper" in n.lower() for n in names], dtype=bool)
        if grip.any() and not grip.all():
            return ~grip
    mask = np.ones(int(action_dim), dtype=bool)
    if action_dim == 16:
        mask[-2:] = False
    elif action_dim >= 7:
        mask[-1] = False
    return mask


def _as_bool_mask(joint_mask: np.ndarray | torch.Tensor, like: np.ndarray | torch.Tensor):
    if torch.is_tensor(like):
        return torch.as_tensor(joint_mask, device=like.device, dtype=torch.bool)
    return np.asarray(joint_mask, dtype=bool)


def _broadcast_q(q_now: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor):
    q = q_now
    if torch.is_tensor(actions):
        q = torch.as_tensor(q, device=actions.device, dtype=actions.dtype)
        while q.ndim < actions.ndim:
            q = q.unsqueeze(-2)
        return q
    q = np.asarray(q, dtype=np.float32)
    while q.ndim < np.asarray(actions).ndim:
        q = np.expand_dims(q, axis=-2)
    return q


def subtract_joint_pose(
    actions: np.ndarray,
    q_now: np.ndarray,
    joint_mask: np.ndarray,
) -> np.ndarray:
    """actions[..., joints] -= q_now[..., joints]. Grippers unchanged."""
    out = np.array(actions, dtype=np.float32, copy=True)
    mask = np.asarray(joint_mask, dtype=bool)
    q = _broadcast_q(q_now, out)
    out[..., mask] -= q[..., mask]
    return out


def add_joint_pose(
    actions: np.ndarray | torch.Tensor,
    q_now: np.ndarray | torch.Tensor,
    joint_mask: np.ndarray,
) -> np.ndarray | torch.Tensor:
    """actions[..., joints] += q_now[..., joints]. Grippers unchanged."""
    mask = _as_bool_mask(joint_mask, actions)
    q = _broadcast_q(q_now, actions)
    if torch.is_tensor(actions):
        out = actions.clone()
        out[..., mask] = out[..., mask] + q[..., mask]
        return out
    out = np.array(actions, dtype=np.float32, copy=True)
    out[..., mask] = out[..., mask] + q[..., mask]
    return out


def stats_predict_joint_delta(stats: dict | None) -> bool:
    """True if action stats were overlaid with joint-delta residuals."""
    if not stats:
        return False
    return "action_delta_mean" in stats or "action_delta_std" in stats


def flow_history_from_phys(
    state_phys: np.ndarray,
    stats: dict,
    norm_mode: str,
    *,
    predict_joint_delta: bool | None = None,
    joint_mask: np.ndarray | None = None,
    action_names: list[str] | None = None,
) -> np.ndarray:
    """Physical state window ``(T, D)`` → normalized A2A flow source.

    Last row is ``q_now``. With joint-delta: joints become ``state - q_now``,
    then **action** normalization (same space as residual action targets).
    Otherwise: state normalization (same as ``obs_state``).
    """
    state_phys = np.asarray(state_phys, dtype=np.float32)
    if predict_joint_delta is None:
        predict_joint_delta = stats_predict_joint_delta(stats)
    if not predict_joint_delta:
        return normalize(state_phys, stats, prefix="state", mode=norm_mode)
    if joint_mask is None:
        joint_mask = joint_mask_from_names(action_names, int(state_phys.shape[-1]))
    hist = subtract_joint_pose(state_phys, state_phys[-1], joint_mask)
    return normalize(hist, stats, prefix="action", mode=norm_mode)


def overlay_joint_delta_action_stats(
    stats: dict[str, np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    *,
    horizon: int,
    joint_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replace ``action_{mean,std,min,max}`` with chunk-relative joint-delta stats.

    For each t, target[k] = action[t+k] − state[t] on joints (k < horizon).
    Mutates ``stats`` in place and returns it.
    """
    if not states or not actions:
        raise ValueError("overlay_joint_delta_action_stats: empty state/action lists")
    if int(horizon) <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")

    if "action_delta_mean" not in stats:
        for key in ("mean", "std", "min", "max"):
            src = f"action_{key}"
            if src in stats:
                stats[f"action_abs_{key}"] = np.asarray(stats[src], dtype=np.float32).copy()

    chunks: list[np.ndarray] = []
    mask = np.asarray(joint_mask, dtype=bool)
    for st, ac in zip(states, actions):
        st = np.asarray(st, dtype=np.float32)
        ac = np.asarray(ac, dtype=np.float32)
        t_len = int(min(st.shape[0], ac.shape[0]))
        for h in range(int(horizon)):
            n = t_len - h
            if n <= 0:
                break
            delta = ac[h : h + n].copy()
            delta[:, mask] -= st[:n, mask]
            chunks.append(delta)
    all_delta = np.concatenate(chunks, axis=0)
    stats["action_mean"] = all_delta.mean(axis=0).astype(np.float32)
    stats["action_std"] = (all_delta.std(axis=0) + 1e-6).astype(np.float32)
    stats["action_min"] = all_delta.min(axis=0).astype(np.float32)
    stats["action_max"] = all_delta.max(axis=0).astype(np.float32)
    stats["action_delta_mean"] = stats["action_mean"].copy()
    stats["action_delta_std"] = stats["action_std"].copy()
    return stats


def denormalize_predicted_action(
    pred_norm: np.ndarray | torch.Tensor,
    stats: dict[str, np.ndarray],
    norm_mode: str,
    *,
    q_now_phys: np.ndarray | torch.Tensor | None,
    predict_joint_delta: bool,
    joint_mask: np.ndarray,
) -> np.ndarray | torch.Tensor:
    """Denormalize policy output; if delta mode, add current joints back."""
    pred = denormalize(pred_norm, stats, prefix="action", mode=norm_mode)
    if not predict_joint_delta:
        return pred
    if q_now_phys is None:
        raise ValueError("predict_joint_delta requires q_now_phys (current joint pose)")
    return add_joint_pose(pred, q_now_phys, joint_mask)
