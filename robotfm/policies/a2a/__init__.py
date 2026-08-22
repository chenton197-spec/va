"""A2A / N-A2A / A2A-U Action-to-Action flow matching policies."""

from typing import Any

from robotfm.policies.a2a.a2a_policy import A2AConfig, A2APolicy

__all__ = ["A2AConfig", "A2APolicy", "A2AUConfig", "A2AUPolicy"]


def __getattr__(name: str) -> Any:
    if name in {"A2AUConfig", "A2AUPolicy"}:
        from robotfm.policies.a2a.a2a_u_policy import A2AUConfig, A2AUPolicy

        return A2AUConfig if name == "A2AUConfig" else A2AUPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
