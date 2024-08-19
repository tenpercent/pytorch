# flake8: noqa: B950
# fmt: off
# This file was generated by AutoHeuristic. Do not modify it manually!
# To regenerate this file, take a look at the steps in the README.md file inside torchgen/_autoheuristic/mm/
from typing import List, Optional, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)


class MMRankingA100(LearnedHeuristicDecision):

    def __init__(self) -> None:
        self.choices: List[Choice] = []
        self.fill_choices()

    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == 166912
            and str(metadata.device_capa) == "(8, 0)"
        )

    def get_confidence_threshold(self) -> float:
        return 0.0

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
        self.choices.append('extern_mm')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')

    def get_name(self) -> str:
        return 'mm'

    def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:
        if context.get_value('arith_intensity') <= 52.6245059967041:
            if context.get_value('n') <= 34.0:
                if context.get_value('n') <= 18.0:
                    if context.get_value('k*n') <= 312.0:
                        return [(0.093, 12), (0.081, 16), (0.081, 148), (0.070, 10), (0.070, 17), (0.070, 149), (0.070, 151), (0.070, 150), (0.070, 14), (0.058, 11), (0.058, 15), (0.058, 13), (0.058, 122), (0.047, 121), (0.035, 123), (0.012, 92)]
                    else:
                        if context.get_value('k') <= 40.0:
                            return [(0.083, 42), (0.083, 46), (0.083, 44), (0.083, 40), (0.083, 128), (0.067, 45), (0.067, 43), (0.067, 41), (0.067, 169), (0.067, 171), (0.067, 168), (0.067, 129), (0.067, 170), (0.033, 103), (0.017, 121)]
                        else:
                            return [(0.112, 137), (0.104, 136), (0.101, 0), (0.081, 1), (0.073, 135), (0.069, 67), (0.066, 187), (0.058, 41), (0.050, 71), (0.046, 68), (0.046, 70), (0.031, 44), (0.027, 43), (0.027, 170), (0.019, 189), (0.019, 188), (0.015, 169), (0.015, 171), (0.012, 115), (0.012, 168), (0.012, 69), (0.004, 103)]
                else:
                    if context.get_value('mat1_stride_0') <= 20.0:
                        return [(0.069, 0), (0.059, 157), (0.059, 22), (0.059, 153), (0.059, 155), (0.059, 25), (0.059, 23), (0.059, 19), (0.044, 21), (0.044, 18), (0.044, 152), (0.044, 158), (0.044, 154), (0.044, 156), (0.044, 20), (0.044, 124), (0.044, 24), (0.030, 125), (0.029, 126), (0.015, 97), (0.015, 95), (0.015, 96), (0.010, 2), (0.010, 75)]
                    else:
                        if context.get_value('k') <= 68.0:
                            return [(0.087, 72), (0.087, 74), (0.087, 73), (0.086, 76), (0.077, 75), (0.067, 192), (0.058, 190), (0.048, 47), (0.048, 193), (0.048, 49), (0.048, 51), (0.048, 191), (0.038, 53), (0.019, 133), (0.019, 50), (0.019, 175), (0.019, 172), (0.019, 48), (0.019, 174), (0.010, 173), (0.010, 177), (0.010, 52), (0.010, 54), (0.010, 178), (0.010, 176)]
                        else:
                            return [(0.154, 52), (0.154, 72), (0.102, 75), (0.087, 49), (0.087, 73), (0.086, 51), (0.057, 176), (0.045, 2), (0.038, 191), (0.038, 178), (0.038, 190), (0.029, 173), (0.029, 76), (0.026, 138), (0.013, 139), (0.013, 140), (0.003, 0)]
            else:
                if context.get_value('k') <= 35.0:
                    if context.get_value('k') <= 18.0:
                        if context.get_value('m*n') <= 19505152.0:
                            return [(0.151, 159), (0.140, 160), (0.129, 164), (0.055, 127), (0.051, 29), (0.044, 161), (0.044, 147), (0.040, 146), (0.040, 31), (0.037, 145), (0.026, 28), (0.022, 90), (0.022, 93), (0.022, 94), (0.022, 100), (0.022, 125), (0.022, 158), (0.022, 157), (0.011, 87), (0.011, 88), (0.011, 89), (0.011, 91), (0.011, 95), (0.011, 96), (0.011, 98), (0.011, 99)]
                        else:
                            return [(0.069, 7), (0.069, 5), (0.067, 147), (0.066, 8), (0.061, 145), (0.058, 146), (0.052, 124), (0.049, 29), (0.049, 159), (0.046, 31), (0.043, 157), (0.041, 9), (0.041, 4), (0.040, 6), (0.035, 164), (0.035, 160), (0.026, 158), (0.017, 125), (0.017, 28), (0.017, 32), (0.017, 162), (0.017, 27), (0.017, 30), (0.017, 161), (0.009, 33), (0.009, 26), (0.009, 163), (0.006, 0)]
                    else:
                        if context.get_value('n') <= 68.0:
                            return [(0.101, 182), (0.101, 59), (0.088, 57), (0.076, 184), (0.076, 61), (0.076, 179), (0.076, 62), (0.076, 58), (0.063, 180), (0.063, 60), (0.051, 56), (0.050, 181), (0.025, 130), (0.025, 177), (0.025, 183), (0.013, 178), (0.013, 55)]
                        else:
                            return [(0.089, 180), (0.079, 60), (0.066, 35), (0.066, 181), (0.066, 38), (0.066, 58), (0.066, 179), (0.066, 57), (0.062, 184), (0.053, 37), (0.044, 166), (0.040, 55), (0.040, 39), (0.040, 36), (0.040, 165), (0.040, 167), (0.027, 177), (0.027, 34), (0.022, 159)]
                else:
                    if context.get_value('m*n') <= 309760.0:
                        return [(0.298, 0), (0.097, 140), (0.080, 83), (0.072, 86), (0.044, 84), (0.036, 178), (0.036, 117), (0.036, 82), (0.032, 120), (0.032, 85), (0.028, 119), (0.024, 130), (0.024, 109), (0.020, 108), (0.020, 118), (0.012, 104), (0.012, 116), (0.012, 141), (0.012, 144), (0.008, 105), (0.008, 106), (0.008, 111), (0.008, 114), (0.008, 107), (0.008, 132), (0.004, 101), (0.004, 102), (0.004, 110), (0.004, 112), (0.004, 113), (0.004, 131)]
                    else:
                        if context.get_value('n') <= 72.0:
                            return [(0.227, 77), (0.118, 78), (0.102, 194), (0.086, 80), (0.059, 57), (0.054, 81), (0.049, 196), (0.048, 197), (0.048, 59), (0.043, 79), (0.032, 195), (0.027, 180), (0.022, 3), (0.021, 141), (0.016, 60), (0.016, 142), (0.011, 183), (0.011, 0), (0.011, 144)]
                        else:
                            return [(0.140, 186), (0.132, 185), (0.109, 63), (0.085, 65), (0.078, 37), (0.077, 35), (0.062, 197), (0.047, 194), (0.046, 165), (0.046, 57), (0.039, 78), (0.039, 79), (0.039, 66), (0.039, 64), (0.016, 195), (0.008, 159)]
        else:
            if str(context.get_value('using_tf32')) != 'False':
                if context.get_value('m*n') <= 815360.0:
                    if context.get_value('k') <= 1184.0:
                        return [(0.218, 140), (0.205, 0), (0.154, 144), (0.115, 141), (0.051, 185), (0.051, 104), (0.039, 78), (0.038, 116), (0.026, 165), (0.026, 130), (0.026, 178), (0.013, 57), (0.013, 195), (0.013, 167), (0.013, 186)]
                    else:
                        return [(0.901, 0), (0.030, 144), (0.030, 134), (0.016, 3), (0.006, 78), (0.006, 77), (0.002, 57), (0.002, 194), (0.002, 59), (0.002, 60), (0.002, 143)]
                else:
                    if context.get_value('arith_intensity') <= 187.23922729492188:
                        if context.get_value('mat1_stride_0') <= 198.0:
                            return [(0.273, 63), (0.158, 37), (0.152, 35), (0.127, 57), (0.097, 165), (0.053, 185), (0.031, 0), (0.028, 64), (0.014, 60), (0.014, 78), (0.009, 55), (0.008, 134), (0.005, 34), (0.005, 167), (0.005, 179), (0.005, 65), (0.005, 66), (0.005, 186), (0.005, 194), (0.002, 166)]
                        else:
                            return [(0.296, 63), (0.235, 0), (0.132, 64), (0.074, 37), (0.069, 78), (0.051, 185), (0.051, 35), (0.030, 57), (0.020, 77), (0.016, 194), (0.008, 66), (0.007, 65), (0.003, 3), (0.003, 165), (0.003, 141), (0.001, 134), (0.001, 166)]
                    else:
                        return [(0.405, 0), (0.246, 37), (0.177, 63), (0.145, 35), (0.005, 185), (0.005, 65), (0.005, 64), (0.004, 57), (0.003, 66), (0.002, 165), (0.001, 78), (0.001, 55)]
            else:
                return [(0.357, 0), (0.112, 165), (0.101, 57), (0.094, 179), (0.086, 64), (0.074, 167), (0.067, 60), (0.064, 159), (0.033, 35), (0.007, 195), (0.002, 180), (0.001, 34), (0.001, 166), (0.001, 78)]