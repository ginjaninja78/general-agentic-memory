# -*- coding: utf-8 -*-
"""数据集加载和适配模块"""

from benchmarks.datasets.hotpotqa import HotpotQABenchmark
from benchmarks.datasets.narrativeqa import NarrativeQABenchmark
from benchmarks.datasets.locomo import LoCoMoBenchmark
from benchmarks.datasets.ruler import RULERBenchmark

__all__ = [
    "HotpotQABenchmark",
    "NarrativeQABenchmark", 
    "LoCoMoBenchmark",
    "RULERBenchmark",
]

