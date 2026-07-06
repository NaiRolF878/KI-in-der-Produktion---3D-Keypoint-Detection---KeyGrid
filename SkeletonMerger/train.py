# -*- coding: utf-8 -*-
"""
Adapted for KeypointNet dataset (.pcd files)
Category: Airplane (02691156)
"""
import os
import random
import argparse
import contextlib
import torch
import torch.optim as optim
import numpy as np
import open3d as o3d
from merger.merger_net import Net
from merger.composed_chamfer import composed_sqrt_chamfer

arg_parser = argparse.ArgumentParser(description="Training Skeleton Merger on KeypointNet dataset.")
arg_parser.add_argument('--dataset-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\02691156',
                        help='Path to the pcd folder.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='model/sm_airplane.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=80,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Number of points per point cloud.')


def load_pcd_files(folder, max_points=2048):
    point_clouds = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
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


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


def feed(net, optimizer, x_set, train, shuffle, batch, epoch):
    running_loss = 0.0
    running_lrc = 0.0
    running_ldiv = 0.0
    net.train(train)
    if shuffle:
        x_set = list(x_set)
        random.shuffle(x_set)
        x_set = np.array(x_set)
    with contextlib.suppress() if train else torch.no_grad():
        for i in range(len(x_set) // batch):
            idx = slice(i * batch, (i + 1) * batch)
            batch_x = torch.FloatTensor(x_set[idx]).to(next(net.parameters()).device)
            if train:
                optimizer.zero_grad()
            RPCD, KPCD, KPA, LF, MA = net(batch_x)
            blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
            bldiv = L2(LF)
            loss = blrc + bldiv
            if train:
                loss.backward()
                optimizer.step()
            running_lrc += blrc.item()
            running_ldiv += bldiv.item()
            running_loss += loss.item()
            print('[%s%d, %4d] loss: %.4f Lrc: %.4f Ldiv: %.4f' %
                  ('VT'[train], epoch, i,
                   running_loss / (i + 1),
                   running_lrc / (i + 1),
                   running_ldiv / (i + 1)))
    return running_loss / max(i + 1, 1), running_lrc / max(i + 1, 1), running_ldiv / max(i + 1, 1)


if __name__ == '__main__':
    ns = arg_parser.parse_args()

    os.makedirs(os.path.dirname(ns.checkpoint_path) if os.path.dirname(ns.checkpoint_path) else '.', exist_ok=True)

    print(f"Loading point clouds from: {ns.dataset_root}")
    x = load_pcd_files(ns.dataset_root, ns.max_points)
    print(f"Loaded {len(x)} point clouds, shape: {x.shape}")

    if len(x) == 0:
        print("ERROR: No point clouds loaded. Check your --dataset-root path.")
        exit(1)

    # 70/15/15 split
    np.random.shuffle(x)
    split_train = int(len(x) * 0.70)
    split_val   = int(len(x) * 0.85)
    x_train = x[:split_train]
    x_val   = x[split_train:split_val]
    x_test  = x[split_val:]
    print(f"Train: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test)}")

    net = Net(ns.max_points, ns.n_keypoint).to(ns.device)
    optimizer = optim.Adadelta(net.parameters(), eps=1e-2)

    for epoch in range(ns.epochs):
        feed(net, optimizer, x_train, True, True, ns.batch, epoch)
        feed(net, optimizer, x_val, False, False, ns.batch, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, ns.checkpoint_path)
        print(f"Checkpoint saved: {ns.checkpoint_path}")
