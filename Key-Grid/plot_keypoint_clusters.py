# -*- coding: utf-8 -*-
"""
plot_keypoint_clusters.py — Keypoint-Positionen ueber alle Modelle, in mehreren
Projektionen plus echter 3D-Ansicht, mit Schwerpunkt-Markern und einer
Konsistenz-Tabelle (Std-Abweichung pro Keypoint je Achse).

Erweitert den Scatter aus evaluate_airplane_v2.py:
  - XY (Aufsicht), XZ (Seitenansicht), YZ (Frontansicht)
  - kombinierte 3D-Ansicht
  - grosser Schwerpunkt-Marker je Keypoint (zeigt, wo jeder KP "wohnt")
  - Konsistenz-Tabelle: Std je Achse + Gesamt-Streuung pro KP
  - schreibt zusaetzlich keypoint_consistency.json fuer den Report

Zwei Modi:

A) Aus vorhandener Keypoint-Datei (z.B. results/airplane_v2_keypoints.npy):
   python plot_keypoint_clusters.py --keypoints results\\airplane_v2_keypoints.npy

B) Keypoints selbst vorhersagen (wie evaluate_airplane_v2.py), reproduzierbar:
   python plot_keypoint_clusters.py ^
     --dataset-root C:\\Users\\xrstu\\KeypointNet\\pcds\\02691156 ^
     --checkpoint-path model\\airplane_v2.pt ^
     --n-keypoint 8 --test-split-only

Output: keypoint_clusters.png  +  keypoint_consistency.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (aktiviert 3D-Projektion)


# ── kraeftige, gut unterscheidbare Palette ───────────────────────────────────
KP_COLORS = [
    '#ff3b3b', '#ff9f1c', '#ffe600', '#2ecc40',
    '#00e5d4', '#3b82ff', '#c724ff', '#ff5fd2',
    '#a3ff5f', '#5fb0ff', '#ff8a8a', '#b088ff',
]


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#aaaaaa')
    ax.set_ylabel(ylabel, color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.grid(True, color='#2a2a4a', linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='datalim')


def style_ax3d(ax, title):
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('X', color='#aaaaaa')
    ax.set_ylabel('Y', color='#aaaaaa')
    ax.set_zlabel('Z', color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    ax.xaxis.set_pane_color((0.09, 0.13, 0.24, 1.0))
    ax.yaxis.set_pane_color((0.09, 0.13, 0.24, 1.0))
    ax.zaxis.set_pane_color((0.09, 0.13, 0.24, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid']['color'] = (0.16, 0.16, 0.29, 1.0)


def scatter_proj(ax, kp, centers, a, b, n_kp):
    """2D-Projektion mit Punktwolke + Schwerpunkt-Marker. a,b aus {0:X,1:Y,2:Z}."""
    for k in range(n_kp):
        c = KP_COLORS[k % len(KP_COLORS)]
        ax.scatter(kp[:, k, a], kp[:, k, b], c=c, s=3, alpha=0.30,
                   edgecolors='none', zorder=2)
    # Schwerpunkte oben drauf, damit sie trotz Ueberlappung sichtbar sind
    for k in range(n_kp):
        c = KP_COLORS[k % len(KP_COLORS)]
        ax.scatter(centers[k, a], centers[k, b], c=c, s=140,
                   edgecolors='white', linewidths=1.4, zorder=4)
        ax.scatter(centers[k, a], centers[k, b], c='white', s=12, zorder=5)


# ----------------------------------------------------------------------------
# Konsistenz-Tabelle als eigener Subplot
# ----------------------------------------------------------------------------
def draw_consistency_table(ax, stats, n_kp):
    ax.set_facecolor('#16213e')
    ax.axis('off')
    ax.set_title('Konsistenz pro Keypoint  (Std, niedriger = stabiler)',
                 color='white', fontsize=11, fontweight='bold', pad=8)

    header = ['KP', 'std X', 'std Y', 'std Z', 'gesamt']
    col_x = [0.04, 0.30, 0.46, 0.62, 0.80]
    y = 0.90
    for cx, h in zip(col_x, header):
        ax.text(cx, y, h, transform=ax.transAxes, color='#cccccc',
                fontsize=10, fontweight='bold', va='top')
    y -= 0.04
    ax.plot([0.02, 0.98], [y, y], transform=ax.transAxes,
            color='#444466', linewidth=1)
    y -= 0.05

    row_h = 0.82 / max(n_kp, 1)
    for k in range(n_kp):
        c = KP_COLORS[k % len(KP_COLORS)]
        s = stats[k]
        # Farbpunkt + KP-Nummer
        ax.scatter(col_x[0] + 0.02, y - 0.012, c=c, s=70,
                   edgecolors='white', linewidths=0.8,
                   transform=ax.transAxes, clip_on=False)
        ax.text(col_x[0] + 0.06, y, f'{k+1}', transform=ax.transAxes,
                color='white', fontsize=10, va='top')
        vals = [s['std_x'], s['std_y'], s['std_z'], s['std_total']]
        for cx, v in zip(col_x[1:], vals):
            ax.text(cx, y, f'{v:.4f}', transform=ax.transAxes,
                    color='#e0e0e0', fontsize=10, va='top')
        y -= row_h


# ----------------------------------------------------------------------------
# Keypoints beschaffen
# ----------------------------------------------------------------------------
def load_keypoints_from_file(path):
    kp = np.load(path)
    print(f"  Keypoints geladen: {kp.shape}")
    return kp


def predict_keypoints(dataset_root, checkpoint, n_kp, max_points,
                      device, batch, test_split_only):
    import torch
    import open3d as o3d
    import merger.merger_net as merger_net

    def load_pcd(folder, mp):
        clouds = []
        files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
        for fn in files:
            pts = np.asarray(o3d.io.read_point_cloud(os.path.join(folder, fn)).points)
            if len(pts) == 0:
                continue
            if len(pts) >= mp:
                idx = np.random.choice(len(pts), mp, replace=False)
            else:
                idx = np.random.choice(len(pts), mp, replace=True)
            clouds.append(pts[idx])
        return np.array(clouds, dtype=np.float32)

    np.random.seed(42)
    x = load_pcd(dataset_root, max_points)
    np.random.shuffle(x)
    if test_split_only:
        x = x[int(len(x) * 0.85):]
    print(f"  {len(x)} Point Clouds fuer Vorhersage")

    net = merger_net.Net(max_points, n_kp).to(device)
    ckpt = torch.load(checkpoint, map_location=torch.device(device), weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            b = torch.FloatTensor(x[i:i+batch]).to(device)
            kp, _ = net(b, 'False')
            for k in kp:
                out.append(k.cpu().numpy())
    return np.array(out)


def compute_stats(kp, n_kp):
    """Pro Keypoint: Schwerpunkt + Std je Achse + Gesamtstreuung."""
    centers = kp.mean(axis=0)            # (K,3)
    stds = kp.std(axis=0)                # (K,3)
    stats = []
    for k in range(n_kp):
        sx, sy, sz = stds[k]
        stats.append({
            'kp': k + 1,
            'center': centers[k].round(4).tolist(),
            'std_x': float(sx), 'std_y': float(sy), 'std_z': float(sz),
            # Gesamtstreuung = mittlerer 3D-Abstand vom Schwerpunkt
            'std_total': float(np.sqrt(np.mean(np.sum((kp[:, k] - centers[k])**2, axis=1)))),
        })
    return centers, stats


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Keypoint-Cluster in mehreren Ansichten")
    ap.add_argument('--keypoints', type=str, default=None,
                    help='Vorhandene .npy (models x K x 3). Wenn gesetzt, kein Predict.')
    ap.add_argument('--dataset-root', type=str,
                    default=r'C:\Users\xrstu\KeypointNet\pcds\02691156')
    ap.add_argument('--checkpoint-path', type=str, default=r'model\airplane_v2.pt')
    ap.add_argument('--n-keypoint', type=int, default=8)
    ap.add_argument('--max-points', type=int, default=2048)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--test-split-only', action='store_true')
    ap.add_argument('--output', type=str, default='keypoint_clusters.png')
    ap.add_argument('--stats-json', type=str, default='keypoint_consistency.json')
    ap.add_argument('--title', type=str, default='Key-Grid Airplane V2 — Keypoint-Cluster')
    args = ap.parse_args()

    if args.keypoints:
        kp = load_keypoints_from_file(args.keypoints)
    else:
        kp = predict_keypoints(args.dataset_root, args.checkpoint_path,
                               args.n_keypoint, args.max_points,
                               args.device, args.batch, args.test_split_only)

    n_kp = kp.shape[1]
    centers, stats = compute_stats(kp, n_kp)

    # ── Stats in Konsole + JSON ────────────────────────────────────────────
    print(f"\n  {'KP':>3} | {'std X':>7} {'std Y':>7} {'std Z':>7} | {'gesamt':>7}")
    print("  " + "-" * 42)
    for s in stats:
        print(f"  {s['kp']:>3} | {s['std_x']:>7.4f} {s['std_y']:>7.4f} "
              f"{s['std_z']:>7.4f} | {s['std_total']:>7.4f}")
    overall = float(np.mean([s['std_total'] for s in stats]))
    print("  " + "-" * 42)
    print(f"  Mittlere Gesamt-Streuung: {overall:.4f}")
    most = min(stats, key=lambda s: s['std_total'])
    least = max(stats, key=lambda s: s['std_total'])
    print(f"  Kompaktester KP: {most['kp']} ({most['std_total']:.4f})  |  "
          f"Streuendster KP: {least['kp']} ({least['std_total']:.4f})")

    with open(args.stats_json, 'w') as f:
        json.dump({'n_models': int(kp.shape[0]), 'n_keypoints': n_kp,
                   'mean_total_std': overall, 'per_keypoint': stats}, f, indent=2)
    print(f"  Stats gespeichert: {args.stats_json}")

    # ── Figure: 3 Projektionen + Tabelle (oben), 3D (rechts gross) ─────────
    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.32, hspace=0.30,
                           width_ratios=[1, 1, 1, 1.25],
                           height_ratios=[1, 1])

    ax_xy = fig.add_subplot(gs[0, 0])
    scatter_proj(ax_xy, kp, centers, 0, 1, n_kp)
    style_ax(ax_xy, 'XY (Aufsicht)', 'X', 'Y')

    ax_xz = fig.add_subplot(gs[0, 1])
    scatter_proj(ax_xz, kp, centers, 0, 2, n_kp)
    style_ax(ax_xz, 'XZ (Seitenansicht)', 'X', 'Z')

    ax_yz = fig.add_subplot(gs[0, 2])
    scatter_proj(ax_yz, kp, centers, 1, 2, n_kp)
    style_ax(ax_yz, 'YZ (Frontansicht)', 'Y', 'Z')

    # 3D ueber beide Zeilen rechts
    ax3d = fig.add_subplot(gs[:, 3], projection='3d')
    ax3d.set_facecolor('#1a1a2e')
    for k in range(n_kp):
        c = KP_COLORS[k % len(KP_COLORS)]
        ax3d.scatter(kp[:, k, 0], kp[:, k, 1], kp[:, k, 2],
                     c=c, s=4, alpha=0.30, edgecolors='none', label=f'KP{k+1}')
        ax3d.scatter(*centers[k], c=c, s=130, edgecolors='white',
                     linewidths=1.4, depthshade=False)
    style_ax3d(ax3d, '3D-Ansicht (mit Schwerpunkten)')
    ax3d.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white',
                fontsize=8, markerscale=2, loc='upper left',
                bbox_to_anchor=(1.02, 1.0))
    ax3d.view_init(elev=22, azim=-58)

    # Konsistenz-Tabelle unten links (ueber die ersten 3 Spalten)
    ax_tab = fig.add_subplot(gs[1, 0:3])
    draw_consistency_table(ax_tab, stats, n_kp)

    fig.suptitle(args.title, color='white', fontsize=16, fontweight='bold', y=0.97)
    plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Plot gespeichert: {args.output}")


if __name__ == '__main__':
    main()