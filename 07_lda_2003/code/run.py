#!/usr/bin/env python3
"""Runnable companion example for: Latent Dirichlet Allocation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_paper_examples import run_example


if __name__ == "__main__":
    print(run_example("07"))
