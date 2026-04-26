#!/usr/bin/env python
"""CLI entry point for the evaluation pipeline.

Usage::

    python evaluation/scripts/evaluate.py --config evaluation/configs/truckscenes_eval.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure the project root is on the path so ``evaluation`` and ``radargen`` are importable.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.config import EvaluationConfig
from evaluation.evaluator import Evaluator


def _setup_logging(output_dir: str) -> None:
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(output_dir, 'logs', f'evaluation_{ts}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger(__name__).info(f"Logging to {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate point cloud generation models.')
    parser.add_argument('--config', required=True, help='Path to YAML evaluation config.')
    args = parser.parse_args()

    config = EvaluationConfig.from_yaml(args.config)
    _setup_logging(config.output_dir)

    logger = logging.getLogger(__name__)
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Dataset: {config.dataset_name}, split: {config.eval_split}")
    logger.info(f"Models: {[m.name for m in config.models]}")

    evaluator = Evaluator(config)
    results = evaluator.evaluate()
    evaluator.print_summary(results)


if __name__ == '__main__':
    main()
