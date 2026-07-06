# -*- coding: utf-8 -*-
"""
Adapted for KeypointNet dataset (.pcd files)
Mit Wandb (optional) und lokalem Logging
"""
import os
import random
import time
import json
import argparse
import contextlib
import torch
import torch.optim as optim
import numpy as np
import open3d as o3d
from merger.merger_net import Net
from merger.composed_chamfer import loss_all

# ── Wandb optional ─────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

arg_parser = argparse.ArgumentParser(description="Training Key_Grid on KeypointNet dataset.")
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='model/airplane.pt')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=8)
arg_parser.add_argument('-b', '--batch', type=int, default=8)
arg_parser.add_argument('-e', '--epochs', type=int, default=100)
arg_parser.add_argument('--max-points', type=int, default=2048)
arg_parser.add_argument("--keynumber", type=int, default=12)
arg_parser.add_argument("--chamfer", type=int, default=20)
arg_parser.add_argument("--lambda_init_points", type=float, default=1.0)
arg_parser.add_argument("--lambda_chamfer", type=float, default=1.0)
arg_parser.add_argument('--dataset-root', type=str, default=r'C:\Users\xrstu\KeypointNet\pcds\02691156')
arg_parser.add_argument('--wandb-project', type=str, default='keypoint-detection')
arg_parser.add_argument('--no-wandb', action='store_true')


def load_pcd_files(folder, max_points=2048):
    point_clouds = []
    files = [f for f in os.listdir(folder) if f.endswith('.pcd')]
    print(f"Found {len(files)} .pcd files in {folder}")
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            pcd = o3d.io.read_point_cloud(fpath)
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                continue
            if len(pts) >= max_points:
                idx = np.random.choice(len(pts), max_points, replace=False)
            else:
                idx = np.random.choice(len(pts), max_points, replace=True)
            point_clouds.append(pts[idx])
        except Exception as e:
            print(f"Could not load {fname}: {e}")
    return np.array(point_clouds, dtype=np.float32)


def feed(net, optimizer, x_set, train, shuffle, batch, epoch, ns):
    running_init_points = 0.0
    running_chamfer = 0.0
    net.train(train)
    if shuffle:
        x_set = list(x_set)
        random.shuffle(x_set)
        x_set = np.array(x_set)
    n_batches = len(x_set) // batch
    with contextlib.suppress() if train else torch.no_grad():
        for i in range(n_batches):
            idx = slice(i * batch, (i + 1) * batch)
            batch_x = torch.FloatTensor(x_set[idx]).cuda()
            if train:
                optimizer.zero_grad()
            keypoint, reconstruct = net(batch_x, 'True')
            loss = loss_all(batch_x, keypoint, reconstruct, epoch, ns)
            running_init_points += loss['init_points']
            if epoch > ns.chamfer:
                running_chamfer += loss['chamfer']
            total_loss = sum(loss.values())
            if train:
                total_loss.backward()
                optimizer.step()
    train_init = running_init_points / max(n_batches, 1)
    train_chamfer = running_chamfer / max(n_batches, 1)
    return train_init, train_chamfer, train_init + train_chamfer


if __name__ == '__main__':
    ns = arg_parser.parse_args()

    os.makedirs(os.path.dirname(ns.checkpoint_path) if os.path.dirname(ns.checkpoint_path) else '.', exist_ok=True)

    # ── Wandb ──────────────────────────────────────────────────────────────
    USE_WANDB = WANDB_AVAILABLE and not ns.no_wandb
    wandb_run = None
    if USE_WANDB:
        try:
            wandb.login(anonymous='never', timeout=10)
            wandb_run = wandb.init(
                project=ns.wandb_project,
                name=f'KeyGrid-{os.path.basename(ns.dataset_root)}-{time.strftime("%H%M")}',
                config={
                    'model': 'Key-Grid',
                    'epochs': ns.epochs,
                    'batch': ns.batch,
                    'n_keypoint': ns.n_keypoint,
                    'dataset': ns.dataset_root,
                    'chamfer_start': ns.chamfer,
                }
            )
            print("  [wandb] Eingeloggt.")
        except Exception:
            USE_WANDB = False
            print("  [INFO] Wandb nicht eingeloggt – nur lokales Logging.")

    # ── Lokales Log ────────────────────────────────────────────────────────
    log_path = os.path.join(
        os.path.dirname(ns.checkpoint_path) if os.path.dirname(ns.checkpoint_path) else '.',
        'training_log.json'
    )
    local_log = {
        'config': {
            'epochs': ns.epochs, 'batch': ns.batch,
            'n_keypoint': ns.n_keypoint, 'dataset': ns.dataset_root,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'epochs': []
    }

    # ── Daten laden ────────────────────────────────────────────────────────
    print(f"Loading point clouds from: {ns.dataset_root}")
    x = load_pcd_files(ns.dataset_root, ns.max_points)
    print(f"Loaded {len(x)} point clouds, shape: {x.shape}")

    if len(x) == 0:
        print("ERROR: No point clouds loaded. Check your --dataset-root path.")
        exit(1)

    np.random.shuffle(x)
    split_train = int(len(x) * 0.70)
    split_val   = int(len(x) * 0.85)
    x_train = x[:split_train]
    x_val   = x[split_train:split_val]
    x_test  = x[split_val:]
    print(f"Train: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test)}")

    local_log['config']['n_train'] = len(x_train)
    local_log['config']['n_val']   = len(x_val)
    local_log['config']['n_test']  = len(x_test)

    # ── Training ───────────────────────────────────────────────────────────
    net = Net(ns.max_points, ns.n_keypoint).cuda()
    optimizer = optim.Adadelta(net.parameters(), lr=0.1, eps=1e-2)

    for epoch in range(ns.epochs):
        t0 = time.time()

        train_init, train_chamfer, train_loss = feed(net, optimizer, x_train, True,  True,  ns.batch, epoch, ns)
        val_init,   val_chamfer,   val_loss   = feed(net, optimizer, x_val,   False, False, ns.batch, epoch, ns)

        epoch_time = time.time() - t0

        # Lokales Log
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss), 'train_init': float(train_init), 'train_chamfer': float(train_chamfer),
            'val_loss':   float(val_loss),   'val_init':   float(val_init),   'val_chamfer':   float(val_chamfer),
            'epoch_time': float(epoch_time),
            'timestamp':  time.strftime('%H:%M:%S')
        }
        local_log['epochs'].append(entry)
        with open(log_path, 'w') as f:
            json.dump(local_log, f, indent=2)

        # Wandb Log
        if wandb_run is not None:
            wandb_run.log({
                'train/loss': train_loss,
                'train/init_points': train_init,
                'train/chamfer': train_chamfer,
                'val/loss': val_loss,
                'val/init_points': val_init,
                'val/chamfer': val_chamfer,
                'epoch_time': epoch_time,
                'epoch': epoch
            })

        print(f"[Ep {epoch:3d}] "
              f"Train Loss: {train_loss:.4f} (init: {train_init:.4f} chamfer: {train_chamfer:.4f}) | "
              f"Val Loss: {val_loss:.4f} | "
              f"Zeit: {epoch_time:.1f}s")

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, ns.checkpoint_path)

    print(f"\nCheckpoint: {ns.checkpoint_path}")
    print(f"Log:        {log_path}")

    if wandb_run is not None:
        wandb_run.finish()