#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real Time Chunking (RTC) and Bidirectional Decoding (BID) configuration classes.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
- Training-time RTC: https://arxiv.org/pdf/2512.05964
"""

from dataclasses import dataclass
from enum import Enum

from lerobot.configs.types import RTCAttentionSchedule, RTCTrainingDelayDistribution


class RTCMode(str, Enum):
    """RTC operation mode."""

    INFERENCE = "inference"  # Inference-time RTC (original)
    TRAINING = "training"  # Training-time RTC (action prefix conditioning)


@dataclass
class RTCConfig:
    """Unified configuration for Real Time Chunking (RTC).

    Supports both inference-time and training-time RTC modes.
    Only one mode can be active at a time.

    Inference-time RTC:
        Improves real-time inference by treating chunk generation as an inpainting problem,
        strategically handling overlapping timesteps between action chunks using prefix attention.

    Training-time RTC:
        Trains the model to condition on action chunks during training, enabling efficient
        real-time action generation at deployment without expensive inference-time resampling.
    """

    # Infrastructure
    enabled: bool = False
    mode: RTCMode = RTCMode.INFERENCE

    # Inference-time RTC settings
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # Training-time RTC settings
    min_delay: int = 0
    max_delay: int = 0
    delay_distribution: RTCTrainingDelayDistribution = RTCTrainingDelayDistribution.UNIFORM
    exp_decay: float = 1.0

    # Debug settings (shared)
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        # Convert string mode to enum if needed
        if isinstance(self.mode, str):
            self.mode = RTCMode(self.mode)

        # Validate inference-time parameters
        if self.mode == RTCMode.INFERENCE and self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")

        # Validate training-time parameters
        if self.mode == RTCMode.TRAINING:
            if self.min_delay < 0:
                raise ValueError(f"min_delay must be >= 0, got {self.min_delay}")
            if self.max_delay < self.min_delay:
                raise ValueError(f"max_delay ({self.max_delay}) must be >= min_delay ({self.min_delay})")
            if self.exp_decay <= 0:
                raise ValueError(f"exp_decay must be positive, got {self.exp_decay}")

        # Validate shared parameters
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")


# Deprecated: Keep for backward compatibility
@dataclass
class RTCTrainingConfig:
    """Deprecated: Use RTCConfig with mode=RTCMode.TRAINING instead.

    This class is kept for backward compatibility but will be removed in a future version.
    """

    enabled: bool = False
    min_delay: int = 0
    max_delay: int = 0
    delay_distribution: RTCTrainingDelayDistribution = RTCTrainingDelayDistribution.UNIFORM
    exp_decay: float = 1.0

    def __post_init__(self):
        if self.min_delay < 0:
            raise ValueError(f"min_delay must be >= 0, got {self.min_delay}")
        if self.max_delay < self.min_delay:
            raise ValueError(f"max_delay ({self.max_delay}) must be >= min_delay ({self.min_delay})")
        if self.exp_decay <= 0:
            raise ValueError(f"exp_decay must be positive, got {self.exp_decay}")

    def to_rtc_config(self) -> RTCConfig:
        """Convert to unified RTCConfig."""
        return RTCConfig(
            enabled=self.enabled,
            mode=RTCMode.TRAINING,
            min_delay=self.min_delay,
            max_delay=self.max_delay,
            delay_distribution=self.delay_distribution,
            exp_decay=self.exp_decay,
        )
