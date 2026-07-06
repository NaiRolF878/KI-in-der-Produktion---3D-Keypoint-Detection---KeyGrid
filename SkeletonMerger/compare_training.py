# -*- coding: utf-8 -*-
"""
Vergleichs-Training: Key-Grid vs Skeleton Merger
Kategorie: Bathtub (02808440)
Loggt Loss, Geschwindigkeit, Konvergenz und erstellt Plots
Wandb ist optional – läuft auch ohne Login, speichert immer lokal
"""
import os
import sys
import random
import time
import json
import argparse
import contextlib
import numpy as np
import torch
import torch.optim as optim
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Wandb optional einbinden ───────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("  [INFO] wandb nicht installiert – nur lokales Logging aktiv.")

# ── Argumente ──────────────────────────────────────────────────────────────
arg_parser = argparse.ArgumentParser(description="Vergleichs-Training Key-Grid vs Skeleton Merger")
arg_parser.add_argument('--dataset-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\02808440',
                        help='Pfad zum Bathtub pcd Ordner.')
arg_parser.add_argument('--keygrid-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid',
                        help='Pfad zum Key-Grid Repo.')
arg_parser.add_argument('--skeletonmerger-dir', type=str,
                        default=r'C:\Users\xrstu\SkeletonMerger',
                        help='Pfad zum SkeletonMerger Repo.')
arg_parser.add_argument('--epochs', type=int, default=80,
                        help='Anzahl Epochen.')
arg_parser.add_argument('--batch', type=int, default=16,
                        help='Batch Size.')
arg_parser.add_argument('--n-keypoint', type=int, default=8,
                        help='Anzahl Keypoints.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Punkte pro Point Cloud.')
arg_parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Ausgabeordner fuer Plots und Logs.')
arg_parser.add_argument('--wandb-project', type=str, default='keypoint-detection',
                        help='Wandb Projektname.')
arg_parser.add_argument('--no-wandb', action='store_true',
                        help='Wandb deaktivieren auch wenn installiert.')
ns = arg_parser.parse_args()

os.makedirs(ns.output_dir, exist_ok=True)

# ── Wandb initialisieren ───────────────────────────────────────────────────
USE_WANDB = WANDB_AVAILABLE and not ns.no_wandb

if USE_WANDB:
    try:
        wandb.login(anonymous='never', timeout=10)
        USE_WANDB = True
        print("  [wandb] Eingeloggt – Metriken werden in der Cloud gespeichert.")
    except Exception:
        USE_WANDB = False
        print("  [INFO] Wandb nicht eingeloggt – nur lokales Logging aktiv.")

# ── Lokales Logging ────────────────────────────────────────────────────────
log_path = os.path.join(ns.output_dir, 'training_log.json')
local_log = {
    'config': {
        'epochs': ns.epochs,
        'batch': ns.batch,
        'n_keypoint': ns.n_keypoint,
        'max_points': ns.max_points,
        'dataset': ns.dataset_root,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    },
    'keygrid': [],
    'skeletonmerger': []
}

def log_epoch(model_name, epoch, train_loss, val_loss, epoch_time, wandb_run=None):
    entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch_time': epoch_time,
        'timestamp': time.strftime('%H:%M:%S')
    }
    local_log[model_name].append(entry)
    with open(log_path, 'w') as f:
        json.dump(local_log, f, indent=2)
    if wandb_run is not None:
        wandb_run.log({
            f'{model_name}/train_loss': train_loss,
            f'{model_name}/val_loss': val_loss,
            f'{model_name}/epoch_time': epoch_time,
            'epoch': epoch
        })

# ── Daten laden ────────────────────────────────────────────────────────────
def load_pcd_files(folder, max_points=2048):
    point_clouds = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
    print(f"  Gefunden: {len(files)} .pcd Dateien")
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
        except:
            pass
    return np.array(point_clouds, dtype=np.float32)

print("\n" + "="*60)
print("  Lade Bathtub Point Clouds...")
print("="*60)
x_all = load_pcd_files(ns.dataset_root, ns.max_points)
np.random.shuffle(x_all)
split_train = int(len(x_all) * 0.70)
split_val   = int(len(x_all) * 0.85)
x_train = x_all[:split_train]
x_val   = x_all[split_train:split_val]
x_test  = x_all[split_val:]
print(f"  Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

local_log['config']['n_train'] = len(x_train)
local_log['config']['n_val']   = len(x_val)
local_log['config']['n_test']  = len(x_test)

metrics = {
    'keygrid':        {'train_loss': [], 'val_loss': [], 'epoch_time': []},
    'skeletonmerger': {'train_loss': [], 'val_loss': [], 'epoch_time': []}
}

# ══════════════════════════════════════════════════════════════════════════
#  KEY-GRID TRAINING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  KEY-GRID Training startet...")
print("="*60)

kg_run = None
if USE_WANDB:
    kg_run = wandb.init(
        project=ns.wandb_project,
        name=f'KeyGrid-Bathtub-{time.strftime("%H%M")}',
        config={'model': 'Key-Grid', 'epochs': ns.epochs, 'batch': ns.batch,
                'n_keypoint': ns.n_keypoint, 'dataset': 'Bathtub'},
        reinit=True
    )

sys.path.insert(0, ns.keygrid_dir)
from merger.merger_net import Net as KGNet
from merger.composed_chamfer import loss_all

class KGArgs:
    keynumber = 12
    chamfer = 20
    lambda_init_points = 1.0
    lambda_chamfer = 1.0
kg_args = KGArgs()
kg_args.batch = ns.batch
kg_args.epochs = ns.epochs

kg_net = KGNet(ns.max_points, ns.n_keypoint).cuda()
kg_optimizer = optim.Adadelta(kg_net.parameters(), lr=0.1, eps=1e-2)

def kg_feed(net, optimizer, x_set, train, batch, epoch):
    running_loss = 0.0
    net.train(train)
    x_set = list(x_set)
    if train:
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
            loss_dict = loss_all(batch_x, keypoint, reconstruct, epoch, kg_args)
            loss = sum(loss_dict.values())
            running_loss += loss.item()
            if train:
                loss.backward()
                optimizer.step()
    return running_loss / max(n_batches, 1)

for epoch in range(ns.epochs):
    t0 = time.time()
    train_loss = kg_feed(kg_net, kg_optimizer, x_train, True, ns.batch, epoch)
    val_loss   = kg_feed(kg_net, kg_optimizer, x_val, False, ns.batch, epoch)
    epoch_time = time.time() - t0
    metrics['keygrid']['train_loss'].append(train_loss)
    metrics['keygrid']['val_loss'].append(val_loss)
    metrics['keygrid']['epoch_time'].append(epoch_time)
    log_epoch('keygrid', epoch, train_loss, val_loss, epoch_time, kg_run)
    print(f"[KG] Epoche {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Zeit: {epoch_time:.1f}s")

torch.save({'epoch': epoch, 'model_state_dict': kg_net.state_dict()},
           os.path.join(ns.output_dir, 'keygrid_bathtub.pt'))
print("  Key-Grid Modell gespeichert.")
if kg_run is not None:
    kg_run.finish()

sys.path.pop(0)
for mod in list(sys.modules.keys()):
    if 'merger' in mod:
        del sys.modules[mod]

# ══════════════════════════════════════════════════════════════════════════
#  SKELETON MERGER TRAINING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SKELETON MERGER Training startet...")
print("="*60)

sm_run = None
if USE_WANDB:
    sm_run = wandb.init(
        project=ns.wandb_project,
        name=f'SkeletonMerger-Bathtub-{time.strftime("%H%M")}',
        config={'model': 'Skeleton Merger', 'epochs': ns.epochs, 'batch': ns.batch,
                'n_keypoint': ns.n_keypoint, 'dataset': 'Bathtub'},
        reinit=True
    )

sys.path.insert(0, ns.skeletonmerger_dir)
from merger.merger_net import Net as SMNet
from merger.composed_chamfer import composed_sqrt_chamfer

def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))

sm_net = SMNet(ns.max_points, ns.n_keypoint).cuda()
sm_optimizer = optim.Adadelta(sm_net.parameters(), eps=1e-2)

def sm_feed(net, optimizer, x_set, train, batch, epoch):
    running_loss = 0.0
    net.train(train)
    x_set = list(x_set)
    if train:
        random.shuffle(x_set)
    x_set = np.array(x_set)
    n_batches = len(x_set) // batch
    with contextlib.suppress() if train else torch.no_grad():
        for i in range(n_batches):
            idx = slice(i * batch, (i + 1) * batch)
            batch_x = torch.FloatTensor(x_set[idx]).cuda()
            if train:
                optimizer.zero_grad()
            RPCD, KPCD, KPA, LF, MA = net(batch_x)
            blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
            bldiv = L2(LF)
            loss = blrc + bldiv
            running_loss += loss.item()
            if train:
                loss.backward()
                optimizer.step()
    return running_loss / max(n_batches, 1)

for epoch in range(ns.epochs):
    t0 = time.time()
    train_loss = sm_feed(sm_net, sm_optimizer, x_train, True, ns.batch, epoch)
    val_loss   = sm_feed(sm_net, sm_optimizer, x_val, False, ns.batch, epoch)
    epoch_time = time.time() - t0
    metrics['skeletonmerger']['train_loss'].append(train_loss)
    metrics['skeletonmerger']['val_loss'].append(val_loss)
    metrics['skeletonmerger']['epoch_time'].append(epoch_time)
    log_epoch('skeletonmerger', epoch, train_loss, val_loss, epoch_time, sm_run)
    print(f"[SM] Epoche {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Zeit: {epoch_time:.1f}s")

torch.save({'epoch': epoch, 'model_state_dict': sm_net.state_dict()},
           os.path.join(ns.output_dir, 'skeletonmerger_bathtub.pt'))
print("  Skeleton Merger Modell gespeichert.")
if sm_run is not None:
    sm_run.finish()

sys.path.pop(0)

# ── Finale Metriken speichern ──────────────────────────────────────────────
with open(os.path.join(ns.output_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n  Lokales Log:    {log_path}")
print(f"  Metriken JSON:  {os.path.join(ns.output_dir, 'metrics.json')}")

# ══════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Erstelle Vergleichs-Plots...")
print("="*60)

epochs_list = list(range(ns.epochs))
kg = metrics['keygrid']
sm = metrics['skeletonmerger']

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

COLORS = {
    'kg_train': '#00d4ff', 'kg_val': '#0099bb',
    'sm_train': '#ff6b35', 'sm_val': '#cc4400',
}

def style_ax(ax, title):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors='#aaaaaa')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.grid(True, color='#2a2a4a', linestyle='--', alpha=0.6)
    ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white', fontsize=9)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_list, kg['train_loss'], color=COLORS['kg_train'], linewidth=2, label='Key-Grid Train')
ax1.plot(epochs_list, sm['train_loss'], color=COLORS['sm_train'], linewidth=2, label='Skeleton Merger Train')
ax1.set_xlabel('Epoche'); ax1.set_ylabel('Loss')
style_ax(ax1, 'Train Loss Vergleich')

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_list, kg['val_loss'], color=COLORS['kg_val'], linewidth=2, label='Key-Grid Val')
ax2.plot(epochs_list, sm['val_loss'], color=COLORS['sm_val'], linewidth=2, label='Skeleton Merger Val')
ax2.set_xlabel('Epoche'); ax2.set_ylabel('Loss')
style_ax(ax2, 'Validation Loss Vergleich')

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs_list, kg['epoch_time'], color=COLORS['kg_train'], linewidth=2, label='Key-Grid')
ax3.plot(epochs_list, sm['epoch_time'], color=COLORS['sm_train'], linewidth=2, label='Skeleton Merger')
ax3.set_xlabel('Epoche'); ax3.set_ylabel('Sekunden')
style_ax(ax3, 'Zeit pro Epoche')

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(epochs_list, kg['train_loss'], color=COLORS['kg_train'], linewidth=2, label='Train')
ax4.plot(epochs_list, kg['val_loss'],   color=COLORS['kg_val'],   linewidth=2, label='Validation', linestyle='--')
ax4.set_xlabel('Epoche'); ax4.set_ylabel('Loss')
style_ax(ax4, 'Key-Grid: Train vs Val')

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(epochs_list, sm['train_loss'], color=COLORS['sm_train'], linewidth=2, label='Train')
ax5.plot(epochs_list, sm['val_loss'],   color=COLORS['sm_val'],   linewidth=2, label='Validation', linestyle='--')
ax5.set_xlabel('Epoche'); ax5.set_ylabel('Loss')
style_ax(ax5, 'Skeleton Merger: Train vs Val')

ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#16213e')
ax6.axis('off')

def convergence_epoch(losses):
    target = losses[0] * 0.1
    for i, l in enumerate(losses):
        if l <= target:
            return i
    return ns.epochs

summary = [
    ("", "Key-Grid", "Skel.Merger"),
    ("─"*8, "─"*10, "─"*10),
    ("Train Loss", f"{kg['train_loss'][-1]:.4f}", f"{sm['train_loss'][-1]:.4f}"),
    ("Val Loss",   f"{kg['val_loss'][-1]:.4f}",   f"{sm['val_loss'][-1]:.4f}"),
    ("Ø Zeit/Ep.", f"{np.mean(kg['epoch_time']):.1f}s", f"{np.mean(sm['epoch_time']):.1f}s"),
    ("Gesamtzeit", f"{np.sum(kg['epoch_time'])/60:.1f}min", f"{np.sum(sm['epoch_time'])/60:.1f}min"),
    ("Konvergenz", f"Ep. {convergence_epoch(kg['train_loss'])}", f"Ep. {convergence_epoch(sm['train_loss'])}"),
]

y = 0.95
for row in summary:
    ax6.text(0.02, y, row[0], transform=ax6.transAxes, color='#aaaaaa', fontsize=9, va='top')
    ax6.text(0.42, y, row[1], transform=ax6.transAxes, color=COLORS['kg_train'], fontsize=9, va='top', fontweight='bold')
    ax6.text(0.72, y, row[2], transform=ax6.transAxes, color=COLORS['sm_train'], fontsize=9, va='top', fontweight='bold')
    y -= 0.13

ax6.set_title('Zusammenfassung', color='white', fontsize=12, fontweight='bold', pad=10)
for spine in ax6.spines.values():
    spine.set_edgecolor('#444466')

fig.suptitle('Key-Grid vs Skeleton Merger - Bathtub Dataset',
             color='white', fontsize=16, fontweight='bold', y=0.98)

plot_path = os.path.join(ns.output_dir, 'comparison_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"  Plot gespeichert: {plot_path}")

print("\n" + "="*60)
print("  FERTIG! Ergebnisse in:", ns.output_dir)
if USE_WANDB:
    print("  Wandb Dashboard:  https://wandb.ai")
print("="*60)