"""Locations and small IO helpers shared by the evaluation code.

Every path comes from ``config.yaml`` through ``bcg_config``; nothing here
stores an absolute path.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcg_config import paths

# Reconstructed crack-free layouts, their targets and the trained network.
TRAIN = paths.train_masks
TEST = paths.test_masks
RUN = paths.unet_run
OUT = paths.prior_evaluation

# The five fixed path seeds. Each image and method is sampled at all of them.
SEEDS = [41001, 41002, 41003, 41004, 41005]


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write_json(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_mask(data, folder, name):
    return np.array(Image.open(Path(data) / folder / name)) > 0
