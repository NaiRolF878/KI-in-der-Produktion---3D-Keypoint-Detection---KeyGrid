# -*- coding: utf-8 -*-
"""
Test-Evaluierung: Key-Grid vs Skeleton Merger
Kategorie: Bathtub (02808440)
Evaluiert beide Modelle auf dem Testdatensatz und vergleicht die Ergebnisse
"""
import os
import sys
import json
import argparse
import contextlib
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Argumente ──────────────────────────────────────────────────────────────
arg_parser = argparse.ArgumentParser(description="Test-Evaluierung Key-Grid vs Skeleton Merger")
arg_parser.add_argument('--dataset-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\02808440',
                        help='Pfad zum Bathtub pcd Ordner.')
arg_parser.add_argument('--keygrid-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid',
                        help='Pfad zum Key-Grid Repo.')
arg_parser.add_argument('--skeletonmerger-dir', type=str,
                        default=r'C:\Users\xrstu\SkeletonMerger',
                        help='Pfad zum SkeletonMerger Repo.')
arg_parser.add_argument('--keygrid-model', type=str,
                        default=r'comparison_results\keygrid_bathtub.pt',
                        help='Pfad zum Key-Grid Modell.')
arg_parser.add_argument('--skeletonmerger-model', type=str,
                        default=r'comparison_results\skeletonmerger_bathtub.pt',
                        help='Pfad zum Skeleton Merger Modell.')
arg_parser.add_argument('--n-keypoint', type=int, default=8)
arg_parser.add_argument('--max-points', type=int, default=2048)
arg_parser.add_argument('--batch', type=int, default=16)
arg_parser.add_argument('--output-dir', type=str, default='comparison_results')
ns = arg_parser.parse_args()

os.makedirs(ns.output_dir, exist_ok=True)

# ── Daten laden ────────────────────────────────────────────────────────────
def load_pcd_files(folder, max_points=2048):
    point_clouds = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
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
print("  Lade Bathtub Test-Daten...")
print("="*60)
np.random.seed(42)  # gleicher Seed wie beim Training für gleiche Aufteilung
x_all = load_pcd_files(ns.dataset_root, ns.max_points)
np.random.shuffle(x_all)
split_train = int(len(x_all) * 0.70)
split_val   = int(len(x_all) * 0.85)
x_test = x_all[split_val:]
print(f"  Testdaten: {len(x_test)} Point Clouds")

# ── Chamfer Distance Berechnung ────────────────────────────────────────────
def chamfer_distance(pc1, pc2):
    """Berechnet Chamfer Distance zwischen zwei Point Clouds (numpy)"""
    # pc1, pc2: (N, 3)
    diff = pc1[:, None, :] - pc2[None, :, :]  # (N, M, 3)
    dist = np.sum(diff**2, axis=-1)            # (N, M)
    cd = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    return cd

def keypoint_spread(keypoints):
    """Misst wie weit die Keypoints voneinander entfernt sind (Diversität)"""
    dists = []
    for i in range(len(keypoints)):
        for j in range(i+1, len(keypoints)):
            d = np.linalg.norm(keypoints[i] - keypoints[j])
            dists.append(d)
    return np.mean(dists)

def keypoint_consistency(kp_list):
    """Misst wie konsistent Keypoints über verschiedene Modelle sind (Standardabweichung)"""
    kp_array = np.array(kp_list)  # (N, k, 3)
    std_per_keypoint = np.std(kp_array, axis=0)  # (k, 3)
    return np.mean(std_per_keypoint)

# ── Evaluierung ────────────────────────────────────────────────────────────
def evaluate_model(net, x_test, batch, device='cuda'):
    net.eval()
    all_keypoints = []
    all_chamfer = []

    with torch.no_grad():
        for i in range(0, len(x_test), batch):
            batch_x = torch.FloatTensor(x_test[i:i+batch]).to(device)
            try:
                # Key-Grid gibt (keypoint, reconstruct) zurück
                keypoint, reconstruct = net(batch_x, 'False')
                model_type = 'keygrid'
            except TypeError:
                # Skeleton Merger gibt (RPCD, KPCD, KPA, LF, MA) zurück
                reconstruct, keypoint, _, _, _ = net(batch_x)
                model_type = 'skeletonmerger'

            for j in range(len(batch_x)):
                pc = batch_x[j].cpu().numpy()
                kp = keypoint[j].cpu().numpy()
                rc = reconstruct[j].cpu().numpy()
                all_keypoints.append(kp)
                cd = chamfer_distance(pc, rc)
                all_chamfer.append(cd)

    return all_keypoints, all_chamfer

# Key-Grid evaluieren
print("\n" + "="*60)
print("  Evaluiere Key-Grid auf Testdaten...")
print("="*60)
sys.path.insert(0, ns.keygrid_dir)
from merger.merger_net import Net as KGNet
kg_net = KGNet(ns.max_points, ns.n_keypoint).cuda()
kg_net.load_state_dict(torch.load(ns.keygrid_model, map_location='cuda', weights_only=False)['model_state_dict'])
kg_keypoints, kg_chamfer = evaluate_model(kg_net, x_test, ns.batch)
sys.path.pop(0)
for mod in list(sys.modules.keys()):
    if 'merger' in mod:
        del sys.modules[mod]

# Skeleton Merger evaluieren
print("\n" + "="*60)
print("  Evaluiere Skeleton Merger auf Testdaten...")
print("="*60)
sys.path.insert(0, ns.skeletonmerger_dir)
from merger.merger_net import Net as SMNet
sm_net = SMNet(ns.max_points, ns.n_keypoint).cuda()
sm_net.load_state_dict(torch.load(ns.skeletonmerger_model, map_location='cuda', weights_only=False)['model_state_dict'])

sm_all_keypoints = []
sm_all_chamfer = []
sm_net.eval()
with torch.no_grad():
    for i in range(0, len(x_test), ns.batch):
        batch_x = torch.FloatTensor(x_test[i:i+ns.batch]).cuda()
        reconstruct, keypoint, _, _, _ = sm_net(batch_x)
        for j in range(len(batch_x)):
            pc = batch_x[j].cpu().numpy()
            kp = keypoint[j].cpu().numpy()
            rc = reconstruct[j].cpu().numpy()
            sm_all_keypoints.append(kp)
            cd = chamfer_distance(pc, rc)
            sm_all_chamfer.append(cd)

sys.path.pop(0)

# ── Metriken berechnen ─────────────────────────────────────────────────────
kg_mean_cd    = np.mean(kg_chamfer)
kg_std_cd     = np.std(kg_chamfer)
kg_spread     = np.mean([keypoint_spread(kp) for kp in kg_keypoints])
kg_consistency = keypoint_consistency(kg_keypoints)

sm_mean_cd    = np.mean(sm_all_chamfer)
sm_std_cd     = np.std(sm_all_chamfer)
sm_spread     = np.mean([keypoint_spread(kp) for kp in sm_all_keypoints])
sm_consistency = keypoint_consistency(sm_all_keypoints)

print("\n" + "="*60)
print("  TESTERGEBNISSE")
print("="*60)
print(f"{'Metrik':<25} {'Key-Grid':>12} {'Skel.Merger':>12}")
print("-"*50)
print(f"{'Chamfer Distance':.<25} {kg_mean_cd:>12.4f} {sm_mean_cd:>12.4f}")
print(f"{'Chamfer Std':.<25} {kg_std_cd:>12.4f} {sm_std_cd:>12.4f}")
print(f"{'Keypoint Spread':.<25} {kg_spread:>12.4f} {sm_spread:>12.4f}")
print(f"{'Keypoint Konsistenz':.<25} {kg_consistency:>12.4f} {sm_consistency:>12.4f}")
print(f"{'Testmodelle':.<25} {len(kg_chamfer):>12} {len(sm_all_chamfer):>12}")

# Ergebnisse speichern
results = {
    'keygrid': {
        'chamfer_mean': kg_mean_cd, 'chamfer_std': kg_std_cd,
        'keypoint_spread': kg_spread, 'keypoint_consistency': kg_consistency,
        'n_test': len(kg_chamfer)
    },
    'skeletonmerger': {
        'chamfer_mean': sm_mean_cd, 'chamfer_std': sm_std_cd,
        'keypoint_spread': sm_spread, 'keypoint_consistency': sm_consistency,
        'n_test': len(sm_all_chamfer)
    }
}
with open(os.path.join(ns.output_dir, 'test_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

COLORS = {'kg': '#00d4ff', 'sm': '#ff6b35'}

def style_ax(ax, title):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors='#aaaaaa')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.grid(True, color='#2a2a4a', linestyle='--', alpha=0.6)

# 1. Chamfer Distance Verteilung
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(kg_chamfer, bins=20, color=COLORS['kg'], alpha=0.7, label='Key-Grid')
ax1.hist(sm_all_chamfer, bins=20, color=COLORS['sm'], alpha=0.7, label='Skeleton Merger')
ax1.axvline(kg_mean_cd, color=COLORS['kg'], linewidth=2, linestyle='--')
ax1.axvline(sm_mean_cd, color=COLORS['sm'], linewidth=2, linestyle='--')
ax1.set_xlabel('Chamfer Distance')
ax1.set_ylabel('Anzahl Modelle')
ax1.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax1, 'Chamfer Distance Verteilung')

# 2. Balkendiagramm Metriken
ax2 = fig.add_subplot(gs[0, 1])
metriken = ['Chamfer\nDistance', 'Chamfer\nStd', 'Keypoint\nSpread', 'Keypoint\nKonsistenz']
kg_vals = [kg_mean_cd, kg_std_cd, kg_spread, kg_consistency]
sm_vals = [sm_mean_cd, sm_std_cd, sm_spread, sm_consistency]
x = np.arange(len(metriken))
w = 0.35
ax2.bar(x - w/2, kg_vals, w, color=COLORS['kg'], alpha=0.8, label='Key-Grid')
ax2.bar(x + w/2, sm_vals, w, color=COLORS['sm'], alpha=0.8, label='Skeleton Merger')
ax2.set_xticks(x)
ax2.set_xticklabels(metriken, color='#aaaaaa', fontsize=9)
ax2.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax2, 'Metriken Vergleich')

# 3. Chamfer Distance pro Modell (sortiert)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(sorted(kg_chamfer), color=COLORS['kg'], linewidth=1.5, label='Key-Grid')
ax3.plot(sorted(sm_all_chamfer), color=COLORS['sm'], linewidth=1.5, label='Skeleton Merger')
ax3.set_xlabel('Modell (sortiert)')
ax3.set_ylabel('Chamfer Distance')
ax3.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax3, 'Chamfer Distance (sortiert)')

# 4. Zusammenfassung
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#16213e')
ax4.axis('off')

def winner(kg_val, sm_val, lower_better=True):
    if lower_better:
        return 'Key-Grid' if kg_val < sm_val else 'Skel.Merger'
    else:
        return 'Key-Grid' if kg_val > sm_val else 'Skel.Merger'

rows = [
    ("", "Key-Grid", "Skel.Merger", "Besser"),
    ("─"*8, "─"*9, "─"*10, "─"*10),
    ("Chamfer Mean", f"{kg_mean_cd:.4f}", f"{sm_mean_cd:.4f}", winner(kg_mean_cd, sm_mean_cd)),
    ("Chamfer Std",  f"{kg_std_cd:.4f}",  f"{sm_std_cd:.4f}",  winner(kg_std_cd, sm_std_cd)),
    ("KP Spread",    f"{kg_spread:.4f}",   f"{sm_spread:.4f}",   winner(kg_spread, sm_spread, False)),
    ("KP Konsistenz",f"{kg_consistency:.4f}",f"{sm_consistency:.4f}", winner(kg_consistency, sm_consistency)),
]

y = 0.95
for row in rows:
    ax4.text(0.02, y, row[0], transform=ax4.transAxes, color='#aaaaaa', fontsize=9, va='top')
    ax4.text(0.38, y, row[1], transform=ax4.transAxes, color=COLORS['kg'], fontsize=9, va='top', fontweight='bold')
    ax4.text(0.62, y, row[2], transform=ax4.transAxes, color=COLORS['sm'], fontsize=9, va='top', fontweight='bold')
    if len(row) > 3:
        color = COLORS['kg'] if row[3] == 'Key-Grid' else COLORS['sm']
        ax4.text(0.85, y, row[3], transform=ax4.transAxes, color=color, fontsize=9, va='top', fontweight='bold')
    y -= 0.14

ax4.set_title('Zusammenfassung Testergebnisse', color='white', fontsize=12, fontweight='bold', pad=10)
for spine in ax4.spines.values():
    spine.set_edgecolor('#444466')

fig.suptitle('Key-Grid vs Skeleton Merger – Test Evaluation (Bathtub)',
             color='white', fontsize=15, fontweight='bold', y=0.98)

plot_path = os.path.join(ns.output_dir, 'test_evaluation_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"\n  Plot gespeichert: {plot_path}")
print(f"  JSON gespeichert: {os.path.join(ns.output_dir, 'test_results.json')}")
print("\n" + "="*60)
print("  EVALUIERUNG ABGESCHLOSSEN")
print("="*60)
