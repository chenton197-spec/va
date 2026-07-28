"""Real-Time Chunking (RTC) configuration.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
- LeRobot RTCConfig (ported for robotfm)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting
    problem, strategically handling overlapping timesteps between action chunks
    using prefix attention.
    """

    enabled: bool = True
    prefix_attention_schedule: RTCAttentionSchedule | str = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10
    # Fixed delay (in action steps) used by sync eval to simulate inference latency.
    inference_delay: int = 0
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.prefix_attention_schedule, str):
            key = self.prefix_attention_schedule.upper()
            self.prefix_attention_schedule = RTCAttentionSchedule(key)
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
        if self.inference_delay < 0:
            raise ValueError(f"inference_delay must be non-negative, got {self.inference_delay}")
        if self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
