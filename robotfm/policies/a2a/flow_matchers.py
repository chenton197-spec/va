"""TorchCFM wrappers ported from A2A_Flow_Matching."""

from __future__ import annotations

import numpy as np
import torch
import torchcfm.conditional_flow_matching as cfm


class BaseFlowMatcher:
    def compute_loss(self, model, target, **kwargs):
        raise NotImplementedError

    def sample(self, model, shape, device, num_steps, return_traces=False, **kwargs):
        raise NotImplementedError


class TorchFlowMatcher(BaseFlowMatcher):
    def __init__(self, fm, num_sampling_steps=6):
        super().__init__()
        self.fm = fm
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target, start=None, **kwargs):
        if start is None:
            x0 = torch.randn_like(target)
        else:
            x0 = start
        timestep, xt, ut = self.fm.sample_location_and_conditional_flow(x0, target)
        vt = model(xt, timestep, **kwargs)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {"loss": loss.item()}

    def sample(
        self,
        model,
        shape,
        device,
        num_steps=None,
        return_traces=False,
        start=None,
        **kwargs,
    ):
        if num_steps is None:
            num_steps = self.num_sampling_steps
        if start is None:
            x = torch.randn(shape, device=device)
        else:
            x = start
        dt = 1.0 / num_steps

        if return_traces:
            traj_history = [x]
            vel_history = [np.zeros_like(x.detach().cpu().numpy())]

        for t in range(num_steps):
            timestep = torch.ones(x.shape[0], device=x.device) * (t / num_steps)
            vt = model(x, timestep, **kwargs)
            x = x + vt * dt

            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)
        return x


class ConditionalFlowMatcher(TorchFlowMatcher):
    def __init__(self, num_sampling_steps=6, **kwargs):
        super().__init__(cfm.ConditionalFlowMatcher(**kwargs), num_sampling_steps)


class ExactOptimalTransportConditionalFlowMatcher(TorchFlowMatcher):
    def __init__(self, num_sampling_steps=6, **kwargs):
        super().__init__(
            cfm.ExactOptimalTransportConditionalFlowMatcher(**kwargs),
            num_sampling_steps,
        )
