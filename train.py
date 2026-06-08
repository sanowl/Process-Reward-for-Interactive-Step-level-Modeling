#!/usr/bin/env python
"""Convenience entry point: ``python train.py [--env ...] [--quick]``.

The implementation lives in :mod:`prism.experiment` so it is importable as part
of the installed package (and reachable via the ``prism`` console script). This
thin shim keeps the familiar ``python train.py`` workflow working from a clone.
"""
from __future__ import annotations

from prism.experiment import main

if __name__ == "__main__":
    main()
