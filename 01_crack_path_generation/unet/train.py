"""Train the layout-conditioned crack-probability U-Net.

Saves the best checkpoint by internal-validation loss and a resumable last one.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import UNet


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def atomic_torch(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


class Masks(Dataset):
    def __init__(self, data, names):
        self.data, self.names = Path(data), names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        arrays = []
        for folder in ['brick_binary', 'crack_binary']:
            with Image.open(self.data / folder / name) as image:
                a = np.array(image.convert('L'), dtype=np.uint8)
            if a.shape != (512, 512) or not np.all((a == 0) | (a == 255)):
                raise ValueError(f'Unexpected size/encoding: {folder}/{name}')
            arrays.append(torch.from_numpy(a).float().div_(255))
        brick, target = arrays
        return torch.stack([brick, 1-brick]), target.unsqueeze(0)


def loss_parts(logits, target):
    logits = logits.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = (1 - (2*(prob*target).sum((2, 3))) / ((prob+target).sum((2, 3))+1e-6)).mean()
    return 0.7*bce+0.3*dice, bce, dice


def render_predictions(model, dataset, out, epoch):
    model.eval()
    sheet = Image.new('RGB', (768, 8*278), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, index in enumerate(np.linspace(0, len(dataset)-1, 8, dtype=int)):
        x, target = dataset[index]
        with torch.no_grad(), torch.cuda.amp.autocast():
            prob = model(x[None].cuda()).sigmoid()[0, 0].float().cpu().numpy()
        tiles = [x[0].numpy()*255, target[0].numpy()*255, prob*255]
        draw.text((5, j*278), f'Epoch {epoch} | layout / target / probability | {dataset.names[index][:45]}', fill='black')
        for col, arr in enumerate(tiles):
            image = Image.fromarray(np.rint(arr).astype(np.uint8)).convert('RGB').resize((256, 256))
            sheet.paste(image, (col*256, j*278+22))
    sheet.save(out / 'validation_latest.jpg')


def train(args):
    run = args.run.resolve()
    config = json.loads((run / 'config.json').read_text())
    split = json.loads((run / 'split.json').read_text())
    if hashlib.sha256((run / 'split.json').read_bytes()).hexdigest() != config['split_sha256']:
        raise RuntimeError('Split changed since configuration was prepared.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use the configured virtual environment.')
    torch.set_num_threads(4)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(config['seed'])
    train_ds = Masks(config['data_dir'], split['train'])
    val_ds = Masks(config['data_dir'], split['validation'])
    tr_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                           generator=generator, num_workers=0, pin_memory=True)
    va_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    model = UNet(in_ch=2, base=32, p_drop=0.15).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    environment = dict(torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__,
                       gpu=torch.cuda.get_device_name(0), parameters=params, pid=os.getpid())
    print(json.dumps(environment), flush=True)
    if args.smoke_test:
        x, y = next(iter(tr_loader))
        x, y = x.cuda(), y.cuda()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss, _, _ = loss_parts(logits, y)
        assert logits.shape == y.shape and torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            val_loss, _, _ = loss_parts(model(x), y)
        assert torch.isfinite(val_loss)
        report = dict(status='passed', batch_size=len(x), train_loss=float(loss),
                       post_step_eval_loss=float(val_loss), gpu_peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                       **environment)
        atomic_json(run / 'smoke_test.json', report)
        print(json.dumps(report), flush=True)
        return
    best, best_epoch, start = math.inf, 0, 1
    history = []
    if args.resume:
        ckpt = torch.load(run / 'last.pt', map_location='cuda', weights_only=False)
        # The user may extend a completed run's epoch budget. All other
        # training/data settings must still match the saved checkpoint.
        budget_keys = {'epochs', 'stopping_rule'}
        previous = ckpt['config']
        if ({k: v for k, v in previous.items() if k not in budget_keys} !=
                {k: v for k, v in config.items() if k not in budget_keys}):
            raise RuntimeError('Resume config mismatch outside epoch budget.')
        if config['epochs'] < previous['epochs'] or config['epochs'] <= ckpt['epoch']:
            raise RuntimeError('Resume requires a nondecreasing epoch budget and unfinished epochs.')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        best, best_epoch, start = ckpt['best_val'], ckpt['best_epoch'], ckpt['epoch']+1
        torch.set_rng_state(ckpt['rng_torch'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt['rng_cuda']])
        generator.set_state(ckpt['loader_rng'].cpu())
        history = ckpt['history']
        print(f'RESUMED completed epoch {ckpt["epoch"]}; continuing {start}-{config["epochs"]}; '
              f'best_epoch={best_epoch}, best_val={best:.6f}; optimiser and RNG states restored.', flush=True)
    elif (run / 'last.pt').exists() or (run / 'best.pt').exists():
        raise RuntimeError('Checkpoints already exist. Use --resume to continue this run.')
    atomic_json(run / 'environment.json', environment)
    start_time = time.time()
    for epoch in range(start, config['epochs']+1):
        epoch_start = time.time()
        model.train()
        sums = np.zeros(3)
        seen = 0
        for bi, (x, y) in enumerate(tr_loader, 1):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config['amp']):
                parts = loss_parts(model(x), y)
            if not torch.isfinite(parts[0]):
                raise RuntimeError(f'Nonfinite train loss: epoch {epoch}, batch {bi}')
            scaler.scale(parts[0]).backward()
            scaler.step(opt)
            scaler.update()
            sums += np.array([p.item() for p in parts])*len(x)
            seen += len(x)
            if bi == 1 or bi % 10 == 0 or bi == len(tr_loader):
                status = dict(state='training', pid=os.getpid(), epoch=epoch, max_epochs=config['epochs'],
                              batch=bi, batches=len(tr_loader), images_seen=seen, train_images=len(train_ds),
                              train_loss=float(sums[0]/seen), best_epoch=best_epoch,
                              best_val_loss=best if math.isfinite(best) else None,
                              elapsed_seconds=round(time.time()-start_time, 1), updated_unix=time.time())
                atomic_json(run / 'status.json', status)
                print(f'Epoch {epoch:02d}/{config["epochs"]} batch {bi:03d}/{len(tr_loader)} loss={sums[0]/seen:.6f}', flush=True)
        model.eval()
        vsums = np.zeros(3)
        tp = fp = fn = 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=config['amp']):
                    logits = model(x)
                    parts = loss_parts(logits, y)
                if not torch.isfinite(parts[0]):
                    raise RuntimeError('Nonfinite validation loss.')
                vsums += np.array([p.item() for p in parts])*len(x)
                pred, target = logits > 0, y > 0.5
                tp += int((pred & target).sum())
                fp += int((pred & ~target).sum())
                fn += int((~pred & target).sum())
        val_loss = float(vsums[0]/len(val_ds))
        improved = val_loss < best
        if improved:
            best, best_epoch = val_loss, epoch
        record = dict(epoch=epoch, train_loss=float(sums[0]/seen), val_loss=val_loss,
                      val_bce=float(vsums[1]/len(val_ds)), val_soft_dice_loss=float(vsums[2]/len(val_ds)),
                      val_global_iou_at_0_5=tp/max(tp+fp+fn, 1),
                      val_global_dice_at_0_5=2*tp/max(2*tp+fp+fn, 1),
                      learning_rate=opt.param_groups[0]['lr'], epoch_seconds=time.time()-epoch_start,
                      best_epoch=best_epoch, gpu_peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30)
        history.append(record)
        with (run / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(history)
        model_state = dict(model=model.state_dict(), in_ch=2, img_size=512, use_brick_type=False,
                           base=32, p_drop=0.15, mask_encoding=config['mask_encoding'], epoch=epoch,
                           best_val=best, best_epoch=best_epoch, val_loss=val_loss, config=config)
        if improved:
            atomic_torch(run / 'best.pt', model_state)
        atomic_torch(run / 'last.pt', dict(**model_state, optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                     rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                     loader_rng=generator.get_state(), history=history))
        render_predictions(model, val_ds, run, epoch)
        atomic_json(run / 'status.json', dict(state='epoch_complete', pid=os.getpid(), max_epochs=config['epochs'],
                     best_val_loss=best, **record, updated_unix=time.time()))
        print(f'EPOCH COMPLETE {epoch}: train={record["train_loss"]:.6f} val={val_loss:.6f} '
              f'best_epoch={best_epoch} seconds={record["epoch_seconds"]:.1f}', flush=True)
    atomic_json(run / 'status.json', dict(state='completed', pid=os.getpid(), epochs_completed=config['epochs'],
                 best_epoch=best_epoch, best_val_loss=best, elapsed_seconds=time.time()-start_time,
                 stopping_reason=f'Reached configured {config["epochs"]}-epoch maximum; inspect metrics for convergence.', updated_unix=time.time()))
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    try:
        train(args)
    except BaseException as exc:
        atomic_json(args.run / 'status.json', dict(state='failed', pid=os.getpid(), error=str(exc),
                    traceback=traceback.format_exc(), updated_unix=time.time()))
        raise
