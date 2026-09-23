#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Addition8 dynamic uncertainty training with YOLOv8x-seg.

Selection formula in this file: one_minus_mean_conf

Workflow implemented:
  1) Score all images in the Addition6 BCG3 synthetic pool.
  2) Select the top 200 uncertain images and train base + 200 synthetic.
  3) Use stage-1 best.pt to score remaining images; select next 200; train cumulative +400 => model stage 2.
  4) Use stage-2 best.pt to score remaining images; select next 600; train cumulative +1000 => model stage 3.
  5) Use stage-3 best.pt to score remaining images; select next 1000; train cumulative +2000 => model stage 4.

Uncertainty CSVs are written under:
    <result_dir>/merged_data/uncertainty_ranking_<score>__base_<size>__stage_<...>.csv

This file is self-contained and does not import any local training engine.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcg_config import paths

GROUP_NAME = 'addition8_group1'
SOURCE_ADDITION7_GROUP = 'addition7_group1'
JOB_SLUG = 'r200_r400_r600'
RANDOM_SEED = 7101

# Every location comes from config.yaml, under the `downstream:` block.
_DOWNSTREAM = paths.get('downstream') or {}


def _setting(key, default=None):
    """One downstream setting, overridable by the matching environment variable."""
    variable = 'BCG_DOWNSTREAM_' + key.upper()
    if os.environ.get(variable):
        return os.environ[variable]
    value = _DOWNSTREAM.get(key)
    return default if value in (None, '') else value


def _required_dir(key, description):
    value = _setting(key)
    if not value:
        raise SystemExit(
            f'config.yaml does not set downstream.{key} ({description}). '
            f'Copy config.example.yaml and fill it in, or set BCG_DOWNSTREAM_{key.upper()}.'
        )
    return Path(str(value)).expanduser().resolve()


EXPERIMENT_ROOT = _required_dir('experiment_root',
                                'the working directory for the acquisition runs')
ADDITION8_ROOT = EXPERIMENT_ROOT / 'Active_Learning_Addition8'
ADDITION7_ROOT = EXPERIMENT_ROOT / 'Active_Learning_Addition7'
GROUP_ROOT = ADDITION8_ROOT / GROUP_NAME
WORK_ROOT = GROUP_ROOT / 'workspaces' / JOB_SLUG
WORK_ROOT.mkdir(parents=True, exist_ok=True)
(WORK_ROOT / 'yolo_config').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(WORK_ROOT / 'yolo_config')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TQDM_DISABLE'] = '1'
os.environ['ULTRALYTICS_VERBOSE'] = 'False'

# Disable noisy Ultralytics/tqdm batch progress bars in Slurm output.
class NoOpTQDM:
    def __init__(self, iterable=None, *args, **kwargs):
        if iterable is None:
            iterable = kwargs.get('iterable', [])
        self.iterable = iterable
        self.total = kwargs.get('total', None)
        self.n = 0
    def __iter__(self):
        for x in self.iterable:
            yield x
    def __len__(self):
        try:
            return len(self.iterable)
        except Exception:
            return 0
    def update(self, *args, **kwargs): pass
    def close(self): pass
    def set_description(self, *args, **kwargs): pass
    def set_postfix(self, *args, **kwargs): pass
    def refresh(self, *args, **kwargs): pass
    def reset(self, *args, **kwargs): pass
    def clear(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

import sys
import json
import shutil
import random
import gc
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set

import pandas as pd
import numpy as np
import yaml
import torch
from ultralytics import YOLO

import ultralytics.utils
ultralytics.utils.TQDM = NoOpTQDM
try:
    import ultralytics.engine.trainer as trainer_mod
    trainer_mod.TQDM = NoOpTQDM
except Exception:
    pass
try:
    import ultralytics.engine.validator as validator_mod
    validator_mod.TQDM = NoOpTQDM
except Exception:
    pass
from ultralytics.utils import LOGGER
LOGGER.setLevel('ERROR')

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Each concurrent launcher has a private working directory.
os.chdir(WORK_ROOT)
print('Current working directory:', os.getcwd())

# =========================
# Configuration
# =========================
# Reuse the two real-data splits and all real-only benchmark outputs from Addition7.
# Addition8 never repartitions the real data and never trains a benchmark model.
ORIGINAL_ROOT = (ADDITION7_ROOT / SOURCE_ADDITION7_GROUP / 'real_data').resolve()
ROOT = GROUP_ROOT.resolve()

# The BCG synthetic candidate pool the acquisition loop draws from.
SYNTHETIC_POOL_ROOT = _required_dir('synthetic_pool',
                                    'the BCG synthetic candidate pool')
POOL_NAME = SYNTHETIC_POOL_ROOT.name
POOL_SOURCE_CANDIDATES = [SYNTHETIC_POOL_ROOT]

# The fixed 150-image validation partition. It defaults to the dataset export.
_valid = _setting('valid_images')
VALID_IMAGES = (Path(str(_valid)).expanduser().resolve() if _valid
                else paths.dataset_root / 'valid' / 'images')
BASE_TRAIN_SIZES = [200, 400, 600]
BENCHMARK_LINE_SIZES = sorted(set(BASE_TRAIN_SIZES))
BASELINE_RUN_NAME = 'benchmark_train'

MODEL_WEIGHTS = str(_setting('model_weights', 'yolov8x-seg.pt'))
TASK = 'segment'
CLASS_NAMES = ['brick', 'broken_brick', 'crack']
TARGET_CLASS_FOR_SECOND_CURVE = 'crack'

# The manuscript protocol: 60 epochs, batch 8, initial learning rate 0.005.
EPOCHS = int(_setting('epochs', 60))
BATCH = int(_setting('batch', 8))
IMGSZ = int(_setting('image_size', 640))
LR0 = float(_setting('learning_rate', 0.005))
PATIENCE = 0
DEVICE = str(_setting('device', '0'))
WORKERS = int(_setting('workers', 1))

# Dynamic AL schedule: 200 + 200 + 600 + 1000 = 2000 synthetic images.
_addition_points = [200, 400, 1000, 2000, 3000, 4000]
if _addition_points != sorted(set(_addition_points)) or any(x <= 0 for x in _addition_points):
    raise ValueError(f'AL_ADDITION_POINTS must be unique, positive and increasing: {_addition_points}')
ADDITION_STAGES = []
_previous_total = 0
for _stage_idx, _added_total in enumerate(_addition_points, start=1):
    ADDITION_STAGES.append({
        'stage_idx': _stage_idx,
        'added_total': _added_total,
        'increment': _added_total - _previous_total,
        'score_suffix': f'stage_{_stage_idx - 1:03d}_after_add_{_previous_total:05d}_remaining' if _previous_total else 'stage_000_initial_all',
        'score_model_label': 'yolov8x_initial' if _previous_total == 0 else f'model{_stage_idx - 1}_add_{_previous_total:05d}',
    })
    _previous_total = _added_total
CURVE_ADDED_POINTS = [s['added_total'] for s in ADDITION_STAGES]
MIN_REQUIRED_POOL_IMAGES = max(CURVE_ADDED_POINTS)

UNCERTAINTY_SCORE_TYPE = 'one_minus_mean_conf'
UNCERTAINTY_TAG = UNCERTAINTY_SCORE_TYPE.replace('-', '_').replace(' ', '_')
RESULT_DIR_NAME = 'results_r200_r400_r600'

UNCERTAINTY_EMPTY_PREDICTION_SCORE = 1.0
UNCERTAINTY_HIGHER_IS_MORE_UNCERTAIN = True
UNCERTAINTY_PREDICT_BATCH = 1  # memory-safe for Slurm jobs; batch size does not change ranking logic
UNCERTAINTY_PREDICT_CHUNK_SIZE = 128  # run predict in small chunks instead of one 8259-image list
UNCERTAINTY_PREDICT_HALF = True
UNCERTAINTY_PREDICT_IMGSZ = IMGSZ
UNCERTAINTY_PREDICT_CONF = 0.001
UNCERTAINTY_PREDICT_IOU = 0.7
UNCERTAINTY_MAX_DET = 300
REUSE_EXISTING_UNCERTAINTY = True
OVERWRITE_EXISTING_UNCERTAINTY = False

LINK_MODE = 'symlink'
OVERWRITE_EXISTING = False
REUSE_EXISTING_RUNS = True
REUSE_EXISTING_MATERIALIZED_DATA = True
RESET_RESULT_DIR_BEFORE_TRAINING = False

RESUME_INTERRUPTED_RUNS = True
ERROR_ON_INCOMPLETE_RUN_WITHOUT_LAST = True
RECOVER_FINISHED_RUN_WITH_VAL = True

PROGRESS_EVERY_EPOCHS = 2

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print('Current experiment configuration:')
print(f'  GROUP_NAME = {GROUP_NAME}')
print(f'  SOURCE_ADDITION7_GROUP = {SOURCE_ADDITION7_GROUP}')
print(f'  ADDITION8_ROOT = {ADDITION8_ROOT}')
print(f'  ORIGINAL_ROOT (Addition7) = {ORIGINAL_ROOT}')
print(f'  SYNTHETIC_POOL_ROOT (Addition6 BCG3) = {SYNTHETIC_POOL_ROOT}')
print(f'  JOB_SLUG = {JOB_SLUG}')
print(f'  RANDOM_SEED = {RANDOM_SEED}')
print(f'  ROOT = {ROOT}')
print(f'  RESULT_DIR_NAME = {RESULT_DIR_NAME}')
print(f'  UNCERTAINTY_SCORE_TYPE = {UNCERTAINTY_SCORE_TYPE}')
print(f'  BASE_TRAIN_SIZES = {BASE_TRAIN_SIZES}')
print(f'  CURVE_ADDED_POINTS = {CURVE_ADDED_POINTS}')
print(f'  MODEL_WEIGHTS = {MODEL_WEIGHTS}')
print(f'  BATCH = {BATCH}  # override with YOLO_TRAIN_BATCH=16 if needed')
print(f'  WORKERS = {WORKERS}  # override with YOLO_WORKERS=2 if needed')

# =========================
# Logging / utility
# =========================
def log_out(msg):
    print(msg, flush=True)

def log_err(msg):
    print(msg, file=sys.stderr, flush=True)

def make_epoch_progress_callback(stage_name='train'):
    def on_train_epoch_end(trainer):
        current_epoch = int(trainer.epoch) + 1
        total_epochs = int(trainer.epochs)
        if current_epoch % PROGRESS_EVERY_EPOCHS == 0 or current_epoch == total_epochs:
            log_err(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{stage_name}: epoch {current_epoch}/{total_epochs} finished"
            )
    return on_train_epoch_end

def clear_gpu_memory():
    """Aggressively release CUDA/Python objects between uncertainty prediction and training.

    This matters because the same Python process first runs YOLOv8x-seg prediction
    on thousands of pool images and then immediately starts YOLOv8x-seg training.
    Slurm jobs can otherwise fail at the first backward pass with
    CUBLAS_STATUS_ALLOC_FAILED even though the uncertainty CSV was created.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
    time.sleep(1)
    print('GPU cache cleared and garbage collected')

def maybe_cuda_synchronize():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

def log_cuda_memory(prefix='CUDA'):
    try:
        if not torch.cuda.is_available():
            return
        free_b, total_b = torch.cuda.mem_get_info()
        alloc_b = torch.cuda.memory_allocated()
        reserved_b = torch.cuda.memory_reserved()
        gib = 1024 ** 3
        log_err(
            f'[{prefix}] free={free_b/gib:.2f}GiB / total={total_b/gib:.2f}GiB, '
            f'allocated={alloc_b/gib:.2f}GiB, reserved={reserved_b/gib:.2f}GiB'
        )
    except Exception:
        pass

clear_gpu_memory()

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def count_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return 0
    return sum(1 for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTS)

def count_label_files(label_dir):
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return 0
    return sum(1 for p in label_dir.rglob('*.txt') if p.is_file())

def ensure_clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def link_or_copy_file(src, dst, mode='symlink'):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'symlink':
        dst.symlink_to(src)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:
        raise ValueError(f'Unsupported materialization mode: {mode}')

def normalize_rel_key(x) -> str:
    return str(x).replace('\\', '/').lstrip('./')

def pair_key(pair: Dict) -> str:
    return normalize_rel_key(pair['rel'])

def resolve_model_spec(model_like):
    if model_like is None:
        return None
    model_like = str(model_like)
    # Built-in Ultralytics weights such as yolov8x-seg.pt should remain as-is.
    if '/' not in model_like and '\\' not in model_like and not model_like.startswith('.'):
        return model_like
    p = Path(model_like)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return str(p)

def ensure_synthetic_pool_available():
    images_dst = SYNTHETIC_POOL_ROOT / 'images'
    labels_dst = SYNTHETIC_POOL_ROOT / 'labels'
    if images_dst.exists() and labels_dst.exists() and count_images(images_dst) > 0:
        print(f'[pool ready] {SYNTHETIC_POOL_ROOT} images={count_images(images_dst)}')
        return

    SYNTHETIC_POOL_ROOT.mkdir(parents=True, exist_ok=True)
    for src_root in POOL_SOURCE_CANDIDATES:
        src_images = src_root / 'images'
        src_labels = src_root / 'labels'
        if src_images.exists() and src_labels.exists() and count_images(src_images) > 0:
            for sub, src in [('images', src_images), ('labels', src_labels)]:
                dst = SYNTHETIC_POOL_ROOT / sub
                if dst.exists() or dst.is_symlink():
                    continue
                dst.symlink_to(src)
                print(f'[pool symlink] {dst} -> {src}')
            return

    raise FileNotFoundError(
        f'No usable synthetic pool was found. Confirm that images/labels exist under: '
        + ', '.join(str(p) for p in POOL_SOURCE_CANDIDATES)
    )

def collect_yolo_pairs(src_root: Path, source_alias: str) -> List[Dict]:
    src_root = Path(src_root)
    src_images = src_root / 'images'
    src_labels = src_root / 'labels'
    if not src_images.exists():
        raise FileNotFoundError(f'Images directory does not exist: {src_images}')
    if not src_labels.exists():
        raise FileNotFoundError(f'Labels directory does not exist: {src_labels}')

    pairs = []
    for img_path in sorted(src_images.rglob('*')):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
            continue
        rel = img_path.relative_to(src_images)
        label_path = src_labels / rel.with_suffix('.txt')
        if not label_path.exists():
            raise FileNotFoundError(f'Missing label file: {label_path}')
        pairs.append({'img': img_path.resolve(), 'lbl': label_path.resolve(), 'rel': rel, 'source_alias': source_alias})
    return pairs

def infer_num_classes_from_labels(*label_dirs):
    max_class_id = -1
    for label_dir in label_dirs:
        label_dir = Path(label_dir)
        if not label_dir.exists():
            continue
        for txt_file in label_dir.rglob('*.txt'):
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id = int(float(line.split()[0]))
                    max_class_id = max(max_class_id, class_id)
    if max_class_id < 0:
        raise ValueError('No class IDs were found in labels. Check dataset paths and label format.')
    return max_class_id + 1

def get_class_names(nc, class_names=None):
    if class_names is not None:
        if len(class_names) != nc:
            raise ValueError(f'CLASS_NAMES has {len(class_names)} entries, but nc = {nc}')
        return class_names
    return [f'class_{i}' for i in range(nc)]

def prepare_data_yaml(train_images, valid_images, root, save_dir, class_names=None):
    train_images = Path(train_images)
    valid_images = Path(valid_images)
    train_labels = train_images.parent / 'labels'
    valid_labels = valid_images.parent / 'labels'
    required_paths = [train_images, train_labels, valid_images, valid_labels]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Required paths do not exist: ' + ', '.join(missing))
    nc = infer_num_classes_from_labels(train_labels, valid_labels)
    names = get_class_names(nc, class_names)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = {'path': str(root), 'train': str(train_images), 'val': str(valid_images), 'nc': nc, 'names': names}
    yaml_path = save_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    return yaml_path, nc, names

def materialize_dataset(pair_groups: List[Tuple[str, List[Dict]]], dst_dir: Path, mode: str = 'symlink', reuse_existing: bool = True) -> Tuple[Path, Path]:
    dst_dir = Path(dst_dir)
    dst_images = dst_dir / 'images'
    dst_labels = dst_dir / 'labels'
    expected_count = sum(len(pairs) for _, pairs in pair_groups)
    if reuse_existing and dst_images.exists() and dst_labels.exists():
        existing_image_count = count_images(dst_images)
        existing_label_count = count_label_files(dst_labels)
        if existing_image_count == expected_count and existing_label_count == expected_count:
            print(f'[reuse materialized training data] {dst_dir} (images={existing_image_count}, labels={existing_label_count})')
            return dst_images, dst_labels
        print(f'[rebuild training data] count mismatch at {dst_dir}: expected={expected_count}, images={existing_image_count}, labels={existing_label_count}')
    ensure_clean_dir(dst_dir)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    for group_alias, pairs in pair_groups:
        for p in pairs:
            rel = Path(group_alias) / p['rel']
            link_or_copy_file(p['img'], dst_images / rel, mode=mode)
            link_or_copy_file(p['lbl'], dst_labels / rel.with_suffix('.txt'), mode=mode)
    return dst_images, dst_labels

# =========================
# Metrics helpers
# =========================
def extract_metrics(results_dict, task='segment'):
    metrics = {}
    for k, v in results_dict.items():
        try:
            metrics[k] = float(v)
        except Exception:
            metrics[k] = v
    output = {'raw_metrics': metrics}
    output['box_precision'] = metrics.get('metrics/precision(B)')
    output['box_recall'] = metrics.get('metrics/recall(B)')
    output['box_map50'] = metrics.get('metrics/mAP50(B)')
    output['box_map50_95'] = metrics.get('metrics/mAP50-95(B)')
    output['mask_precision'] = metrics.get('metrics/precision(M)')
    output['mask_recall'] = metrics.get('metrics/recall(M)')
    output['mask_map50'] = metrics.get('metrics/mAP50(M)')
    output['mask_map50_95'] = metrics.get('metrics/mAP50-95(M)')
    if task == 'segment':
        output['primary_map50'] = output['mask_map50'] if output['mask_map50'] is not None else output['box_map50']
        output['primary_map50_95'] = output['mask_map50_95'] if output['mask_map50_95'] is not None else output['box_map50_95']
    else:
        output['primary_map50'] = output['box_map50']
        output['primary_map50_95'] = output['box_map50_95']
    return output

def extract_per_class_metrics(metrics_obj, class_names, task='segment'):
    per_class_records = []
    ap_class_index = [int(x) for x in metrics_obj.ap_class_index]
    nt_per_image = getattr(metrics_obj, 'nt_per_image', None)
    nt_per_class = getattr(metrics_obj, 'nt_per_class', None)
    for i, cls_idx in enumerate(ap_class_index):
        vals = [float(x) for x in metrics_obj.class_result(i)]
        row = {
            'class_id': cls_idx,
            'Class': class_names[cls_idx] if class_names is not None else f'class_{cls_idx}',
            'Images': int(nt_per_image[cls_idx]) if nt_per_image is not None else None,
            'Instances': int(nt_per_class[cls_idx]) if nt_per_class is not None else None,
            'Box(P)': vals[0], 'Box(R)': vals[1], 'Box(mAP50)': vals[2], 'Box(mAP50-95)': vals[3],
        }
        if task == 'segment' and len(vals) >= 8:
            row.update({'Mask(P)': vals[4], 'Mask(R)': vals[5], 'Mask(mAP50)': vals[6], 'Mask(mAP50-95)': vals[7]})
        per_class_records.append(row)
    return per_class_records

def get_primary_metric_column(task='segment'):
    return 'Mask(mAP50-95)' if task == 'segment' else 'Box(mAP50-95)'

def get_metric_from_metrics_json(metrics_json_path, task='segment', class_name=None):
    metrics_json_path = Path(metrics_json_path)
    if not metrics_json_path.exists():
        raise FileNotFoundError(f'metrics.json does not exist: {metrics_json_path}')
    with open(metrics_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    summary = data.get('summary', {})
    per_class = data.get('per_class_metrics', [])
    metric_col = get_primary_metric_column(task=task)
    if class_name is None:
        if summary.get('primary_map50_95') is not None:
            return float(summary['primary_map50_95'])
        target_class_name = 'all'
    else:
        target_class_name = str(class_name).strip().lower()
    for row in per_class:
        row_class_name = str(row.get('Class', '')).strip().lower()
        if row_class_name == target_class_name:
            val = row.get(metric_col)
            if val is None and metric_col != 'Box(mAP50-95)':
                val = row.get('Box(mAP50-95)')
            return None if val is None else float(val)
    return None

def get_overall_map50_95(metrics_json_path, task='segment'):
    return get_metric_from_metrics_json(metrics_json_path, task=task, class_name=None)

def get_class_map50_95_from_metrics_json(metrics_json_path, class_name, task='segment'):
    return get_metric_from_metrics_json(metrics_json_path, task=task, class_name=class_name)

def get_class_map50_95_from_records(per_class_records, class_name, task='segment'):
    metric_col = get_primary_metric_column(task)
    for row in per_class_records:
        if str(row.get('Class', '')).strip().lower() == str(class_name).strip().lower():
            val = row.get(metric_col)
            if val is None and metric_col != 'Box(mAP50-95)':
                val = row.get('Box(mAP50-95)')
            return None if val is None else float(val)
    return None

def save_run_outputs(run_dir, size, pool_name, pool_added_images, model_stage, metric_pack, per_class_records, best_model_path, last_model_path, init_weights_path):
    run_dir = Path(run_dir)
    total_row = {
        'class_id': -1,
        'Class': 'all',
        'Images': count_images(VALID_IMAGES),
        'Instances': int(sum(r['Instances'] for r in per_class_records if r['Instances'] is not None)),
        'Box(P)': metric_pack['box_precision'],
        'Box(R)': metric_pack['box_recall'],
        'Box(mAP50)': metric_pack['box_map50'],
        'Box(mAP50-95)': metric_pack['box_map50_95'],
    }
    if TASK == 'segment':
        total_row.update({
            'Mask(P)': metric_pack['mask_precision'],
            'Mask(R)': metric_pack['mask_recall'],
            'Mask(mAP50)': metric_pack['mask_map50'],
            'Mask(mAP50-95)': metric_pack['mask_map50_95'],
        })
    all_records = [total_row] + per_class_records
    per_class_df = pd.DataFrame(all_records)
    per_class_df.insert(0, 'train_size', size)
    per_class_df.insert(1, 'pool_name', pool_name)
    per_class_df.insert(2, 'pool_added_images', pool_added_images)
    per_class_df.insert(3, 'model_stage', model_stage)
    per_class_df.to_csv(run_dir / 'metrics_per_class.csv', index=False)

    target_class_map50_95 = get_class_map50_95_from_records(all_records, TARGET_CLASS_FOR_SECOND_CURVE, task=TASK)
    result_record = {
        'train_size': size,
        'pool_name': pool_name,
        'pool_added_images': pool_added_images,
        'model_stage': model_stage,
        'task': TASK,
        'model_weights': MODEL_WEIGHTS,
        'init_weights_path': str(init_weights_path) if init_weights_path is not None else None,
        'epochs': EPOCHS,
        'batch': BATCH,
        'imgsz': IMGSZ,
        'lr0': LR0,
        'patience': PATIENCE,
        'device': DEVICE,
        'save_dir': str(run_dir),
        'best_model_path': str(best_model_path) if Path(best_model_path).exists() else None,
        'last_model_path': str(last_model_path) if Path(last_model_path).exists() else None,
        'primary_map50': metric_pack['primary_map50'],
        'primary_map50_95': metric_pack['primary_map50_95'],
        f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95': target_class_map50_95,
        'box_map50': metric_pack['box_map50'],
        'box_map50_95': metric_pack['box_map50_95'],
        'mask_map50': metric_pack['mask_map50'],
        'mask_map50_95': metric_pack['mask_map50_95'],
    }
    with open(run_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': result_record, 'per_class_metrics': all_records, 'raw_metrics': metric_pack['raw_metrics']}, f, ensure_ascii=False, indent=2)
    pd.DataFrame([result_record]).to_csv(run_dir / 'metrics.csv', index=False)
    return result_record

# =========================
# Uncertainty scoring / stage selection
# =========================
def compute_uncertainty_from_result(result, score_type: Optional[str] = None) -> Dict:
    if score_type is None:
        score_type = UNCERTAINTY_SCORE_TYPE
    confs = []
    if getattr(result, 'boxes', None) is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    num_det = len(confs)
    if num_det == 0:
        return {
            'num_det': 0,
            'max_conf': None,
            'mean_conf': None,
            'top3_mean_conf': None,
            'uncertainty_score': float(UNCERTAINTY_EMPTY_PREDICTION_SCORE),
        }
    max_conf = float(np.max(confs))
    mean_conf = float(np.mean(confs))
    top3_mean_conf = float(np.mean(sorted(confs, reverse=True)[: min(3, num_det)]))
    if score_type == 'one_minus_max_conf':
        score = 1.0 - max_conf
    elif score_type == 'one_minus_mean_conf':
        score = 1.0 - mean_conf
    elif score_type == 'one_minus_top3_mean_conf':
        score = 1.0 - top3_mean_conf
    else:
        raise ValueError(f'Unsupported UNCERTAINTY_SCORE_TYPE: {score_type}')
    return {
        'num_det': int(num_det),
        'max_conf': max_conf,
        'mean_conf': mean_conf,
        'top3_mean_conf': top3_mean_conf,
        'uncertainty_score': float(score),
    }

def uncertainty_csv_path(curve_root: Path, base_size: int, suffix: str) -> Path:
    # CSVs intentionally live under merged_data as requested, with stage suffixes to avoid overwriting.
    return Path(curve_root) / 'merged_data' / f'uncertainty_ranking_{UNCERTAINTY_TAG}__base_{base_size:04d}__{suffix}.csv'

def selected_order_csv_path(curve_root: Path, base_size: int) -> Path:
    return Path(curve_root) / 'merged_data' / f'selected_order_{UNCERTAINTY_TAG}__base_{base_size:04d}.csv'

def generate_uncertainty_for_pairs(
    pairs: List[Dict],
    model_spec,
    output_csv: Path,
    base_size: int,
    stage_idx: int,
    stage_label: str,
    selected_until_before_scoring: int,
    force: bool = False,
) -> pd.DataFrame:
    """
    Memory-safe + Ultralytics-version-safe uncertainty scoring.

    Do not pass list[str] directly to model.predict(source=...). Some
    Ultralytics versions silently produce no results for list input with
    stream=True. Each chunk is exposed through a temporary symlink directory
    and passed as source=str(chunk_images_dir). Chunking also limits RAM use.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    expected_keys = {pair_key(p) for p in pairs}
    if output_csv.exists() and REUSE_EXISTING_UNCERTAINTY and not force and not OVERWRITE_EXISTING_UNCERTAINTY:
        try:
            old_df = pd.read_csv(output_csv)
            if 'image' in old_df.columns and set(old_df['image'].astype(str)) == expected_keys and len(old_df) == len(expected_keys):
                print(f'[reuse uncertainty CSV] {output_csv} ({len(old_df)} rows)')
                return old_df.sort_values('rank').reset_index(drop=True) if 'rank' in old_df.columns else old_df
            print(f'[rebuild uncertainty CSV] current remaining images do not match {output_csv}')
        except Exception as e:
            print(f'[rebuild uncertainty CSV] failed to read existing CSV: {output_csv} ({e})')

    if not pairs:
        raise ValueError(f'No images are available for scoring: stage={stage_label}, base={base_size}')

    print(
        f'[generate uncertainty] base={base_size}, stage={stage_idx}, label={stage_label}, '
        f'model={model_spec}, images={len(pairs)}, score_type={UNCERTAINTY_SCORE_TYPE}'
    )
    print(
        f'[directory-chunk predict] chunk_size={UNCERTAINTY_PREDICT_CHUNK_SIZE}, '
        f'batch={UNCERTAINTY_PREDICT_BATCH}, half={UNCERTAINTY_PREDICT_HALF}'
    )

    clear_gpu_memory()
    model = YOLO(model_spec)

    rows = []
    scored_keys = set()
    stage_source_root = output_csv.parent / '_uncertainty_predict_sources' / output_csv.stem
    if stage_source_root.exists():
        shutil.rmtree(stage_source_root)
    stage_source_root.mkdir(parents=True, exist_ok=True)

    progress_iter = range(0, len(pairs), UNCERTAINTY_PREDICT_CHUNK_SIZE)
    if tqdm is not None:
        progress_iter = tqdm(
            progress_iter,
            total=(len(pairs) + UNCERTAINTY_PREDICT_CHUNK_SIZE - 1) // UNCERTAINTY_PREDICT_CHUNK_SIZE,
            desc=f'uncertainty chunks base={base_size} {stage_label}',
            unit='chunk',
            ncols=120,
            mininterval=1.0,
        )

    try:
        with torch.inference_mode():
            for start in progress_iter:
                chunk_pairs = pairs[start:start + UNCERTAINTY_PREDICT_CHUNK_SIZE]
                chunk_images_dir = stage_source_root / f'chunk_{start:06d}' / 'images'
                ensure_clean_dir(chunk_images_dir)

                # Map both symlink paths and resolved original-image paths.
                source_abs_to_pair = {}
                source_resolved_to_pair = {}
                rel_to_pair = {}
                for pair in chunk_pairs:
                    rel = Path(pair['rel'])
                    dst_img = chunk_images_dir / rel
                    link_or_copy_file(pair['img'], dst_img, mode='symlink')
                    source_abs_to_pair[str(dst_img.absolute())] = pair
                    source_resolved_to_pair[str(dst_img.resolve())] = pair
                    source_resolved_to_pair[str(Path(pair['img']).resolve())] = pair
                    rel_to_pair[normalize_rel_key(rel)] = pair

                # Pass a directory rather than list[str] for version compatibility.
                results = model.predict(
                    source=str(chunk_images_dir),
                    stream=True,
                    save=False,
                    verbose=False,
                    conf=UNCERTAINTY_PREDICT_CONF,
                    iou=UNCERTAINTY_PREDICT_IOU,
                    imgsz=UNCERTAINTY_PREDICT_IMGSZ,
                    device=DEVICE,
                    max_det=UNCERTAINTY_MAX_DET,
                    batch=UNCERTAINTY_PREDICT_BATCH,
                    half=UNCERTAINTY_PREDICT_HALF,
                    retina_masks=False,
                )

                chunk_result_count = 0
                for result in results:
                    chunk_result_count += 1
                    raw_path = Path(result.path)
                    raw_abs = str(raw_path.absolute())
                    resolved = str(raw_path.resolve())

                    pair = source_abs_to_pair.get(raw_abs) or source_resolved_to_pair.get(resolved)
                    if pair is None:
                        try:
                            rel_from_chunk = normalize_rel_key(raw_path.relative_to(chunk_images_dir))
                            pair = rel_to_pair.get(rel_from_chunk)
                        except Exception:
                            pair = None
                    if pair is None:
                        # Final fallback: match by filename when it is unique.
                        basename = raw_path.name
                        matching = [p for p in chunk_pairs if Path(p['img']).name == basename or Path(p['rel']).name == basename]
                        pair = matching[0] if len(matching) == 1 else None
                    if pair is None:
                        print(f'[warning] result.path cannot be matched to an input image; skipping: {result.path}')
                        continue

                    key = pair_key(pair)
                    if key in scored_keys:
                        continue

                    rel = normalize_rel_key(pair['rel'])
                    stat = compute_uncertainty_from_result(result, score_type=UNCERTAINTY_SCORE_TYPE)
                    rows.append({
                        'base_train_size': base_size,
                        'pool_name': POOL_NAME,
                        'stage_idx': stage_idx,
                        'stage_label': stage_label,
                        'selected_until_before_scoring': selected_until_before_scoring,
                        'remaining_pool_images_before_selection': len(pairs),
                        'image': rel,
                        'filename': Path(rel).name,
                        'image_path': str(Path(pair['img']).resolve()),
                        'label_path': str(Path(pair['lbl']).resolve()),
                        'model_used': str(model_spec),
                        'score_type': UNCERTAINTY_SCORE_TYPE,
                        **stat,
                    })
                    scored_keys.add(key)
                    del result

                if chunk_result_count == 0:
                    raise RuntimeError(
                        'The current chunk produced no YOLO prediction results.\n'
                        f'chunk_images_dir = {chunk_images_dir}\n'
                        'The current Ultralytics version or environment may not have read the source correctly.'
                    )

                del results
                shutil.rmtree(chunk_images_dir.parent, ignore_errors=True)
                clear_gpu_memory()
    finally:
        try:
            shutil.rmtree(stage_source_root, ignore_errors=True)
        except Exception:
            pass
        del model
        clear_gpu_memory()

    if not rows:
        raise RuntimeError(
            'Uncertainty scoring produced no records, so selection and training cannot continue.\n'
            'Check whether model.predict can read the directory-based chunk source.'
        )

    if len(rows) != len(pairs):
        missing = len(pairs) - len(rows)
        example_missing = [pair_key(p) for p in pairs if pair_key(p) not in scored_keys][:10]
        raise RuntimeError(
            f'Scored result count {len(rows)} != input image count {len(pairs)}; missing {missing}.\n'
            f'Example unscored images: {example_missing}\n'
            'Stopped to prevent an invalid selection order. Check image integrity and paths.'
        )

    df = pd.DataFrame(rows)
    if 'uncertainty_score' not in df.columns:
        raise RuntimeError(f'Internal error: uncertainty_score column is missing. columns={list(df.columns)}')

    df = df.sort_values('uncertainty_score', ascending=not UNCERTAINTY_HIGHER_IS_MORE_UNCERTAIN, kind='mergesort').reset_index(drop=True)
    df['rank'] = range(len(df))
    df.to_csv(output_csv, index=False)

    summary = {
        'base_train_size': base_size,
        'pool_name': POOL_NAME,
        'stage_idx': stage_idx,
        'stage_label': stage_label,
        'selected_until_before_scoring': selected_until_before_scoring,
        'remaining_pool_images_before_selection': len(pairs),
        'score_type': UNCERTAINTY_SCORE_TYPE,
        'model_used': str(model_spec),
        'num_rows': int(len(df)),
        'higher_is_more_uncertain': bool(UNCERTAINTY_HIGHER_IS_MORE_UNCERTAIN),
        'output_csv': str(output_csv),
        'top5': df.head(5)[['image', 'uncertainty_score']].to_dict(orient='records') if not df.empty else [],
        'source_mode': 'directory_chunks_symlink',
    }
    with open(output_csv.with_suffix('.summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'[saved uncertainty CSV] {output_csv}')
    print(df.head(5)[['rank', 'image', 'uncertainty_score', 'max_conf', 'num_det']])
    return df

def select_top_pairs_from_uncertainty(df: pd.DataFrame, remaining_by_key: Dict[str, Dict], take_n: int) -> List[Dict]:
    selected = []
    used = set()
    for _, row in df.iterrows():
        key = normalize_rel_key(row['image'])
        pair = remaining_by_key.get(key)
        if pair is None:
            # Fallback by filename if needed; only use it when filename is unique.
            fname = Path(key).name
            matches = [p for k, p in remaining_by_key.items() if Path(k).name == fname]
            if len(matches) == 1:
                pair = matches[0]
                key = pair_key(pair)
        if pair is None or key in used:
            continue
        selected.append(pair)
        used.add(key)
        if len(selected) >= take_n:
            break
    if len(selected) < take_n:
        raise RuntimeError(f'This stage requires {take_n} images, but only {len(selected)} matched the uncertainty CSV.')
    return selected

def save_selected_order(curve_root: Path, base_size: int, selected_pairs: List[Dict], stage_records: List[Dict]):
    rows = []
    first_stage_by_key = {}
    for rec in stage_records:
        for p in rec['pairs']:
            first_stage_by_key[pair_key(p)] = {
                'stage_idx': rec['stage_idx'],
                'added_total': rec['added_total'],
                'increment': rec['increment'],
                'scoring_csv': str(rec['scoring_csv']),
                'scoring_model': str(rec['scoring_model']),
            }
    for idx, p in enumerate(selected_pairs, start=1):
        meta = first_stage_by_key.get(pair_key(p), {})
        rows.append({
            'order_index': idx,
            'base_train_size': base_size,
            'image_rel_path': normalize_rel_key(p['rel']),
            'label_rel_path': normalize_rel_key(Path(p['rel']).with_suffix('.txt')),
            'image_path': str(Path(p['img']).resolve()),
            'label_path': str(Path(p['lbl']).resolve()),
            'first_selected_stage_idx': meta.get('stage_idx'),
            'first_used_at_add': meta.get('added_total'),
            'stage_increment_size': meta.get('increment'),
            'scoring_csv': meta.get('scoring_csv'),
            'scoring_model': meta.get('scoring_model'),
            'score_type': UNCERTAINTY_SCORE_TYPE,
        })
    out = selected_order_csv_path(curve_root, base_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'[saved selected order] {out} ({len(rows)} rows)')
    return out

# =========================
# Training logic
# =========================
def verify_reused_addition7_baselines():
    """Verify Addition7 real splits and benchmarks without creating or training them."""
    for size in BASE_TRAIN_SIZES:
        if size == 0:
            continue
        base_train_dir = ORIGINAL_ROOT / 'train' / str(size)
        train_images = base_train_dir / 'images'
        train_labels = base_train_dir / 'labels'
        run_dir = base_train_dir / BASELINE_RUN_NAME
        metrics_json = run_dir / 'metrics.json'
        best_model_path = run_dir / 'weights' / 'best.pt'

        missing = [
            str(path)
            for path in (train_images, train_labels, metrics_json, best_model_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f'Addition7 R{size} real split or benchmark is incomplete. '
                f'Addition8 will not repartition data or train a replacement benchmark. Missing: {missing}'
            )

        image_count = count_images(train_images)
        label_count = count_label_files(train_labels)
        if image_count != size:
            raise RuntimeError(
                f'Addition7 R{size} expected {size} real images, found {image_count}: {train_images}'
            )
        if label_count != image_count:
            raise RuntimeError(
                f'Addition7 R{size} image/label mismatch: images={image_count}, '
                f'labels={label_count}, labels_dir={train_labels}'
            )

        print(
            f'[reuse Addition7 split + benchmark] R{size}: '
            f'images={image_count}, labels={label_count}, benchmark={run_dir}'
        )

def build_baseline_df() -> pd.DataFrame:
    baseline_records = []
    for size in BENCHMARK_LINE_SIZES:
        baseline_metrics_json = ORIGINAL_ROOT / 'train' / str(size) / BASELINE_RUN_NAME / 'metrics.json'
        if not baseline_metrics_json.exists():
            raise FileNotFoundError(f'Real-data baseline metrics.json was not found: {baseline_metrics_json}')
        baseline_map50_95 = get_overall_map50_95(baseline_metrics_json, task=TASK)
        baseline_target_class_map50_95 = get_class_map50_95_from_metrics_json(baseline_metrics_json, class_name=TARGET_CLASS_FOR_SECOND_CURVE, task=TASK)
        base_train_dir = ORIGINAL_ROOT / 'train' / str(size)
        baseline_records.append({
            'train_size': size,
            'baseline_metrics_json': str(baseline_metrics_json),
            'baseline_overall_map50_95': baseline_map50_95,
            f'baseline_{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95': baseline_target_class_map50_95,
            'baseline_train_images': count_images(base_train_dir / 'images'),
        })
    df = pd.DataFrame(baseline_records).sort_values('train_size').reset_index(drop=True)
    print('Baseline summary from ORIGINAL_ROOT:')
    print(df)
    return df

def train_or_reuse_stage(
    curve_root: Path,
    base_size: int,
    base_pairs: List[Dict],
    selected_pairs: List[Dict],
    added_n: int,
    model_stage: int,
):
    subset_pairs = selected_pairs[:added_n]
    pair_groups = []
    if base_pairs:
        pair_groups.append((f'benchmark_{base_size}', base_pairs))
    pair_groups.append((f'pool_{POOL_NAME}', subset_pairs))

    merged_dir = curve_root / 'merged_data' / f'train_{base_size}__pool_{POOL_NAME}__add_{added_n:05d}'
    merged_images, merged_labels = materialize_dataset(
        pair_groups=pair_groups,
        dst_dir=merged_dir,
        mode=LINK_MODE,
        reuse_existing=REUSE_EXISTING_MATERIALIZED_DATA,
    )

    run_project = curve_root / 'runs' / str(base_size)
    run_name = f'add_{added_n:05d}'
    run_dir = run_project / run_name

    run_dir_existed_before_prepare = run_dir.exists()
    run_dir_nonempty_before_prepare = run_dir_existed_before_prepare and any(run_dir.iterdir())

    if OVERWRITE_EXISTING and run_dir.exists():
        shutil.rmtree(run_dir)
        run_dir_nonempty_before_prepare = False

    yaml_path, nc, names = prepare_data_yaml(
        train_images=merged_images,
        valid_images=Path(VALID_IMAGES).resolve(),
        root=ROOT,
        save_dir=run_dir,
        class_names=CLASS_NAMES,
    )

    metrics_json_path = run_dir / 'metrics.json'
    results_csv_path = run_dir / 'results.csv'
    best_model_path = run_dir / 'weights' / 'best.pt'
    last_model_path = run_dir / 'weights' / 'last.pt'
    stage_init_weights = resolve_model_spec(MODEL_WEIGHTS)

    can_reuse = REUSE_EXISTING_RUNS and metrics_json_path.exists() and best_model_path.exists() and not OVERWRITE_EXISTING

    completed_epochs_before_resume = None
    if results_csv_path.exists():
        try:
            resume_df = pd.read_csv(results_csv_path)
            if 'epoch' in resume_df.columns and not resume_df.empty:
                completed_epochs_before_resume = int(resume_df['epoch'].max()) + 1
            elif not resume_df.empty:
                completed_epochs_before_resume = len(resume_df)
        except Exception:
            completed_epochs_before_resume = None

    appears_finished_without_metrics = (
        RECOVER_FINISHED_RUN_WITH_VAL
        and best_model_path.exists()
        and last_model_path.exists()
        and not metrics_json_path.exists()
        and completed_epochs_before_resume is not None
        and completed_epochs_before_resume >= EPOCHS
        and not OVERWRITE_EXISTING
    )
    can_resume = (
        RESUME_INTERRUPTED_RUNS
        and last_model_path.exists()
        and not metrics_json_path.exists()
        and not appears_finished_without_metrics
        and not OVERWRITE_EXISTING
    )
    incomplete_without_last = (
        RESUME_INTERRUPTED_RUNS
        and run_dir_nonempty_before_prepare
        and not metrics_json_path.exists()
        and not last_model_path.exists()
        and not OVERWRITE_EXISTING
    )

    if can_reuse:
        print(f'[reuse completed result] {metrics_json_path}')
        al_map50_95 = get_overall_map50_95(metrics_json_path, task=TASK)
        target_class_map50_95 = get_class_map50_95_from_metrics_json(metrics_json_path, class_name=TARGET_CLASS_FOR_SECOND_CURVE, task=TASK)
        return {
            'run_dir': run_dir,
            'metrics_json': metrics_json_path,
            'best_model_path': best_model_path,
            'stage_train_images': count_images(merged_images),
            'overall_map50_95': al_map50_95,
            f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95': target_class_map50_95,
            'init_weights_path': stage_init_weights,
        }

    if appears_finished_without_metrics:
        print(f'[recover result] {run_dir} appears to have completed {completed_epochs_before_resume}/{EPOCHS} epochs but lacks metrics.json; validating best.pt.')
        actual_init_weights = str(best_model_path.resolve())
        model = YOLO(actual_init_weights)
        val_results = model.val(data=str(yaml_path), batch=BATCH, imgsz=IMGSZ, device=DEVICE)
        metric_pack = extract_metrics(val_results.results_dict, task=TASK)
        per_class_records = extract_per_class_metrics(val_results, names, task=TASK)
        run_summary = save_run_outputs(run_dir, base_size, POOL_NAME, added_n, model_stage, metric_pack, per_class_records, best_model_path, last_model_path, actual_init_weights)
        try:
            del val_results
            del model
        except Exception:
            pass
        clear_gpu_memory()
        return {
            'run_dir': run_dir,
            'metrics_json': metrics_json_path,
            'best_model_path': best_model_path,
            'stage_train_images': count_images(merged_images),
            'overall_map50_95': float(run_summary['primary_map50_95']),
            f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95': run_summary.get(f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95'),
            'init_weights_path': actual_init_weights,
        }

    actual_init_weights = stage_init_weights
    if can_resume:
        actual_init_weights = str(last_model_path.resolve())
        print(f'[resume training] base={base_size}, add={added_n}, checkpoint={actual_init_weights}')
        if completed_epochs_before_resume is not None:
            print(f'Approximately completed epochs: {completed_epochs_before_resume}/{EPOCHS}')
        clear_gpu_memory()
        maybe_cuda_synchronize()
        log_cuda_memory(prefix=f'before resume train base={base_size} add={added_n}')
        model = YOLO(actual_init_weights)
        model.add_callback('on_train_epoch_end', make_epoch_progress_callback(stage_name=f'resume base={base_size} add={added_n}'))
        train_results = model.train(
            resume=True, batch=BATCH, cache=False, workers=WORKERS, verbose=False,
            seed=RANDOM_SEED, deterministic=True,
        )
    else:
        if incomplete_without_last:
            msg = (
                f'[cannot resume] An incomplete run has no weights/last.pt: {run_dir}\n'
                f'Remove that run directory and retrain, or change the incomplete-run policy explicitly.'
            )
            if ERROR_ON_INCOMPLETE_RUN_WITHOUT_LAST:
                raise RuntimeError(msg)
            print(msg)
        log_out(f'[start training] base={base_size}, add={added_n}, model_stage={model_stage}, train_images={count_images(merged_images)}')
        log_out(f'init weights from: {stage_init_weights}')
        log_out(f'data.yaml: {yaml_path}')
        log_out(f'save_dir: {run_dir}')
        clear_gpu_memory()
        maybe_cuda_synchronize()
        log_cuda_memory(prefix=f'before fresh train base={base_size} add={added_n}')
        model = YOLO(stage_init_weights)
        model.add_callback('on_train_epoch_end', make_epoch_progress_callback(stage_name=f'base={base_size} add={added_n}'))
        train_results = model.train(
            data=str(yaml_path),
            project=str(run_project),
            name=run_name,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            imgsz=IMGSZ,
            lr0=LR0,
            device=DEVICE,
            exist_ok=True,
            cache=False,
            workers=WORKERS,
            verbose=False,
            seed=RANDOM_SEED,
            deterministic=True,
        )

    if train_results is None:
        raise RuntimeError('train() returned no metrics; the training result cannot be recorded.')

    metric_pack = extract_metrics(train_results.results_dict, task=TASK)
    per_class_records = extract_per_class_metrics(train_results, names, task=TASK)
    run_summary = save_run_outputs(run_dir, base_size, POOL_NAME, added_n, model_stage, metric_pack, per_class_records, best_model_path, last_model_path, actual_init_weights)
    log_out(f'[training completed] base={base_size}, add={added_n}, model_stage={model_stage}')
    log_out(f'run_dir: {run_dir}')
    log_out(f'primary_map50_95: {run_summary.get("primary_map50_95")}')

    try:
        del train_results
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    clear_gpu_memory()

    if not best_model_path.exists():
        raise FileNotFoundError(f'best.pt was not found after stage training: {best_model_path}')

    return {
        'run_dir': run_dir,
        'metrics_json': metrics_json_path,
        'best_model_path': best_model_path,
        'stage_train_images': count_images(merged_images),
        'overall_map50_95': float(run_summary['primary_map50_95']),
        f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95': run_summary.get(f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95'),
        'init_weights_path': actual_init_weights,
    }

def run_dynamic_selection_for_base(curve_root: Path, base_size: int, pool_pairs_all: List[Dict], baseline_df: pd.DataFrame):
    if len(pool_pairs_all) < MIN_REQUIRED_POOL_IMAGES:
        raise RuntimeError(f'The synthetic pool has {len(pool_pairs_all)} images, fewer than the required {MIN_REQUIRED_POOL_IMAGES}.')

    if base_size == 0:
        base_pairs = []
        base_count = 0
        baseline_map50_95 = None
        baseline_target_class_map50_95 = None
        baseline_run_dir = None
        baseline_metrics_json = None
        baseline_best_weights = None
    else:
        baseline_row = baseline_df[baseline_df['train_size'] == base_size].iloc[0]
        baseline_map50_95 = float(baseline_row['baseline_overall_map50_95'])
        baseline_target_class_map50_95 = baseline_row.get(f'baseline_{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95')
        if pd.isna(baseline_target_class_map50_95):
            baseline_target_class_map50_95 = None
        else:
            baseline_target_class_map50_95 = float(baseline_target_class_map50_95)
        base_train_dir = ORIGINAL_ROOT / 'train' / str(base_size)
        base_pairs = collect_yolo_pairs(base_train_dir, source_alias=f'train_{base_size}')
        base_count = len(base_pairs)
        baseline_run_dir = ORIGINAL_ROOT / 'train' / str(base_size) / BASELINE_RUN_NAME
        baseline_metrics_json = Path(baseline_row['baseline_metrics_json'])
        baseline_best_weights = baseline_run_dir / 'weights' / 'best.pt'

    print('=' * 100)
    print(f'Starting Addition8 dynamic uncertainty selection: base={base_size}, score={UNCERTAINTY_SCORE_TYPE}')
    print('=' * 100)
    print(f'base train images: {base_count}')
    print(f'pool images: {len(pool_pairs_all)}')

    selected_pairs = []
    selected_keys: Set[str] = set()
    stage_records = []
    combo_curve_rows = []

    if base_size != 0:
        combo_curve_rows.append({
            'train_size': base_size,
            'pool_name': POOL_NAME,
            'pool_added_images': 0,
            'model_stage': 0,
            'total_train_images': base_count,
            'stage_train_images': base_count,
            'overall_map50_95': baseline_map50_95,
            f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95': baseline_target_class_map50_95,
            'gain_vs_baseline': 0.0,
            'rel_gain_vs_baseline_percent': 0.0,
            'run_dir': str(baseline_run_dir),
            'metrics_json': str(baseline_metrics_json),
            'is_baseline': True,
            'init_weights_path': None if baseline_best_weights is None or not baseline_best_weights.exists() else str(baseline_best_weights.resolve()),
            'score_type': UNCERTAINTY_SCORE_TYPE,
        })

    # For a real-data base, the first acquisition must be driven by that
    # baseline model. Using generic COCO weights here would not constitute
    # real-model-guided active selection.
    if baseline_best_weights is not None and baseline_best_weights.exists():
        scoring_model = str(baseline_best_weights.resolve())
    else:
        scoring_model = resolve_model_spec(MODEL_WEIGHTS)

    for stage in ADDITION_STAGES:
        stage_idx = int(stage['stage_idx'])
        added_total = int(stage['added_total'])
        increment = int(stage['increment'])
        score_suffix = stage['score_suffix']

        remaining_pairs = [p for p in pool_pairs_all if pair_key(p) not in selected_keys]
        if len(remaining_pairs) < increment:
            raise RuntimeError(f'base={base_size}, stage={stage_idx}: remaining pool size {len(remaining_pairs)} is below the required increment {increment}.')

        scoring_csv = uncertainty_csv_path(curve_root, base_size, score_suffix)
        score_df = generate_uncertainty_for_pairs(
            pairs=remaining_pairs,
            model_spec=scoring_model,
            output_csv=scoring_csv,
            base_size=base_size,
            stage_idx=stage_idx - 1,  # 0=initial all, 1=after model1, 2=after model2, 3=after model3
            stage_label=score_suffix,
            selected_until_before_scoring=len(selected_pairs),
            force=False,
        )

        remaining_by_key = {pair_key(p): p for p in remaining_pairs}
        new_pairs = select_top_pairs_from_uncertainty(score_df, remaining_by_key, increment)
        selected_pairs.extend(new_pairs)
        selected_keys.update(pair_key(p) for p in new_pairs)

        stage_records.append({
            'stage_idx': stage_idx,
            'added_total': added_total,
            'increment': increment,
            'pairs': new_pairs,
            'scoring_csv': scoring_csv,
            'scoring_model': scoring_model,
        })
        save_selected_order(curve_root, base_size, selected_pairs, stage_records)

        train_info = train_or_reuse_stage(
            curve_root=curve_root,
            base_size=base_size,
            base_pairs=base_pairs,
            selected_pairs=selected_pairs,
            added_n=added_total,
            model_stage=stage_idx,
        )

        al_map50_95 = train_info['overall_map50_95']
        target_class_map50_95 = train_info.get(f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95')
        if baseline_map50_95 is None:
            gain = None
            rel_gain = None
        else:
            gain = al_map50_95 - baseline_map50_95
            rel_gain = (gain / baseline_map50_95 * 100.0) if baseline_map50_95 != 0 else None

        combo_curve_rows.append({
            'train_size': base_size,
            'pool_name': POOL_NAME,
            'pool_added_images': added_total,
            'model_stage': stage_idx,
            'stage_increment_images': increment,
            'total_train_images': base_count + added_total,
            'stage_train_images': train_info['stage_train_images'],
            'overall_map50_95': al_map50_95,
            f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95': target_class_map50_95,
            'gain_vs_baseline': gain,
            'rel_gain_vs_baseline_percent': rel_gain,
            'run_dir': str(train_info['run_dir']),
            'metrics_json': str(train_info['metrics_json']),
            'is_baseline': False,
            'init_weights_path': str(train_info['init_weights_path']),
            'best_model_path': str(train_info['best_model_path']),
            'scoring_csv_used_for_this_stage': str(scoring_csv),
            'scoring_model_used_for_this_stage': str(scoring_model),
            'score_type': UNCERTAINTY_SCORE_TYPE,
        })

        # Next stage scores remaining images with this trained model.
        scoring_model = str(Path(train_info['best_model_path']).resolve())

    combo_curve_df = pd.DataFrame(combo_curve_rows).sort_values('pool_added_images').reset_index(drop=True)
    combo_dir = curve_root / 'summary' / str(base_size)
    combo_dir.mkdir(parents=True, exist_ok=True)
    learning_curve_csv = combo_dir / 'learning_curve.csv'
    combo_curve_df.to_csv(learning_curve_csv, index=False)
    print(f'[saved training records] {learning_curve_csv}')
    print(combo_curve_df[['pool_added_images', 'model_stage', 'total_train_images', 'stage_train_images', 'overall_map50_95', f'{TARGET_CLASS_FOR_SECOND_CURVE}_map50_95', 'gain_vs_baseline']])
    return combo_curve_df

# =========================
# Main
# =========================
def main():
    ensure_synthetic_pool_available()

    print('=' * 90)
    print('[preflight data check] Addition8 dynamic uncertainty')
    print('=' * 90)
    print(f'SYNTHETIC_POOL_ROOT: {SYNTHETIC_POOL_ROOT}')
    print(f'Pool image count: {count_images(SYNTHETIC_POOL_ROOT / "images")}')
    print(f'VALID_IMAGES: {VALID_IMAGES}')
    print(f'BASE_TRAIN_SIZES: {BASE_TRAIN_SIZES}')
    print(f'CURVE_ADDED_POINTS: {CURVE_ADDED_POINTS}')
    print(f'RESULT_DIR_NAME: {RESULT_DIR_NAME}')
    print(f'ADDITION8 OUTPUT ROOT: {ROOT / RESULT_DIR_NAME}')

    curve_root = ROOT / RESULT_DIR_NAME
    if RESET_RESULT_DIR_BEFORE_TRAINING and curve_root.exists():
        print(f'[clear result, train from start] {curve_root}')
        shutil.rmtree(curve_root)
    (curve_root / 'merged_data').mkdir(parents=True, exist_ok=True)
    (curve_root / 'summary').mkdir(parents=True, exist_ok=True)

    pool_pairs_all = collect_yolo_pairs(SYNTHETIC_POOL_ROOT, source_alias=f'pool_{POOL_NAME}')
    pool_pairs_all = sorted(pool_pairs_all, key=pair_key)

    verify_reused_addition7_baselines()
    baseline_df = build_baseline_df()

    all_dfs = []
    for base_size in BASE_TRAIN_SIZES:
        df = run_dynamic_selection_for_base(curve_root, base_size, pool_pairs_all, baseline_df)
        all_dfs.append(df)

    all_curve_df = pd.concat(all_dfs, ignore_index=True, sort=False) if all_dfs else pd.DataFrame()
    all_curve_csv = curve_root / 'summary' / 'all_learning_curves.csv'
    all_curve_df.to_csv(all_curve_csv, index=False)
    print(f'[all completed] {all_curve_csv}')

if __name__ == '__main__':
    main()
