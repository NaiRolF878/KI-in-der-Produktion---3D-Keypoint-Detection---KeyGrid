# -*- coding: utf-8 -*-
"""
Test-Evaluierung: Key-Grid Airplane V2
Evaluiert das Modell auf dem Testdatensatz und erstellt einen detaillierten Report
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

arg_parser = argparse.ArgumentParser(description="Test-Evaluierung Key-Grid Airplane V2")
arg_parser.add_argument('--dataset-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\02691156')
arg_parser.add_argument('--keygrid-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid')
arg_parser.add_argument('--model', type=str,
                        default=r'C:\Users\xrstu\Key-Grid\model\airplane_v2.pt')
arg_parser.add_argument('--n-keypoint', type=int, default=8)
arg_parser.add_argument('--max-points', type=int, default=2048)
arg_parser.add_argument('--batch', type=int, default=16)
arg_parser.add_argument('--output-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid\results\evaluation_airplane_v2')
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
print("  Key-Grid Airplane V2 – Test Evaluierung")
print("="*60)

# Gleicher Seed wie beim Training für gleiche Aufteilung
np.random.seed(42)
print(f"\n  Lade Point Clouds aus: {ns.dataset_root}")
x_all = load_pcd_files(ns.dataset_root, ns.max_points)
np.random.shuffle(x_all)
split_train = int(len(x_all) * 0.70)
split_val   = int(len(x_all) * 0.85)
x_test = x_all[split_val:]
print(f"  Gesamt: {len(x_all)} | Testdaten: {len(x_test)} Point Clouds")

# ── Modell laden ───────────────────────────────────────────────────────────
print(f"\n  Lade Modell: {ns.model}")
sys.path.insert(0, ns.keygrid_dir)
from merger.merger_net import Net

net = Net(ns.max_points, ns.n_keypoint).cuda()
net.load_state_dict(torch.load(ns.model, map_location='cuda', weights_only=False)['model_state_dict'])
net.eval()
print("  Modell geladen.")

# ── Metriken ───────────────────────────────────────────────────────────────
def chamfer_distance(pc1, pc2):
    """Chamfer Distance zwischen zwei Point Clouds"""
    diff = pc1[:, None, :] - pc2[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    cd = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    return float(cd)

def keypoint_spread(keypoints):
    """Durchschnittlicher Abstand zwischen allen Keypoint-Paaren"""
    dists = []
    for i in range(len(keypoints)):
        for j in range(i+1, len(keypoints)):
            d = np.linalg.norm(keypoints[i] - keypoints[j])
            dists.append(d)
    return float(np.mean(dists))

def keypoint_on_surface(keypoints, pointcloud, threshold=0.05):
    """Wie viele Keypoints liegen nah an der Oberfläche?"""
    count = 0
    for kp in keypoints:
        dists = np.linalg.norm(pointcloud - kp, axis=1)
        if np.min(dists) < threshold:
            count += 1
    return count / len(keypoints)

# ── Evaluierung ────────────────────────────────────────────────────────────
print("\n  Evaluiere auf Testdaten...")
all_keypoints = []
all_chamfer = []
all_spread = []
all_surface = []

with torch.no_grad():
    for i in range(0, len(x_test), ns.batch):
        batch_x = torch.FloatTensor(x_test[i:i+ns.batch]).cuda()
        keypoint, reconstruct = net(batch_x, 'False')

        for j in range(len(batch_x)):
            pc  = batch_x[j].cpu().numpy()
            kp  = keypoint[j].cpu().numpy()
            rc  = reconstruct[j].cpu().numpy()

            all_keypoints.append(kp)
            all_chamfer.append(chamfer_distance(pc, rc))
            all_spread.append(keypoint_spread(kp))
            all_surface.append(keypoint_on_surface(kp, pc))

        print(f"  Processed {min(i+ns.batch, len(x_test))}/{len(x_test)}")

# ── Konsistenz berechnen ───────────────────────────────────────────────────
kp_array = np.array(all_keypoints)  # (N, 8, 3)
consistency = float(np.mean(np.std(kp_array, axis=0)))

# ── Ergebnisse zusammenfassen ──────────────────────────────────────────────
results = {
    'model': ns.model,
    'dataset': ns.dataset_root,
    'category': 'Airplane',
    'version': 'v2',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'n_test': len(x_test),
    'n_keypoints': ns.n_keypoint,
    'metrics': {
        'chamfer_mean':    float(np.mean(all_chamfer)),
        'chamfer_std':     float(np.std(all_chamfer)),
        'chamfer_min':     float(np.min(all_chamfer)),
        'chamfer_max':     float(np.max(all_chamfer)),
        'keypoint_spread_mean': float(np.mean(all_spread)),
        'keypoint_spread_std':  float(np.std(all_spread)),
        'keypoint_consistency': consistency,
        'surface_coverage_mean': float(np.mean(all_surface)),
    }
}

# JSON speichern
json_path = os.path.join(ns.output_dir, 'test_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

# Ergebnisse ausgeben
print("\n" + "="*60)
print("  TESTERGEBNISSE – Airplane V2")
print("="*60)
m = results['metrics']
print(f"  Testmodelle:           {len(x_test)}")
print(f"  Chamfer Distance:      {m['chamfer_mean']:.4f} ± {m['chamfer_std']:.4f}")
print(f"  Chamfer Min/Max:       {m['chamfer_min']:.4f} / {m['chamfer_max']:.4f}")
print(f"  Keypoint Spread:       {m['keypoint_spread_mean']:.4f} ± {m['keypoint_spread_std']:.4f}")
print(f"  Keypoint Konsistenz:   {m['keypoint_consistency']:.4f}  (niedriger = besser)")
print(f"  Surface Coverage:      {m['surface_coverage_mean']*100:.1f}%  (Keypoints auf Oberfläche)")
print("="*60)

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

COLOR = '#00d4ff'

def style_ax(ax, title):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#aaaaaa')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.grid(True, color='#2a2a4a', linestyle='--', alpha=0.6)

# 1. Chamfer Distance Verteilung
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(all_chamfer, bins=20, color=COLOR, alpha=0.8, edgecolor='#0099bb')
ax1.axvline(np.mean(all_chamfer), color='#ff6b35', linewidth=2, linestyle='--', label=f'Mean: {np.mean(all_chamfer):.4f}')
ax1.set_xlabel('Chamfer Distance')
ax1.set_ylabel('Anzahl Modelle')
ax1.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax1, 'Chamfer Distance Verteilung')

# 2. Keypoint Spread Verteilung
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(all_spread, bins=20, color='#ff6b35', alpha=0.8, edgecolor='#cc4400')
ax2.axvline(np.mean(all_spread), color=COLOR, linewidth=2, linestyle='--', label=f'Mean: {np.mean(all_spread):.4f}')
ax2.set_xlabel('Keypoint Spread')
ax2.set_ylabel('Anzahl Modelle')
ax2.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax2, 'Keypoint Spread Verteilung')

# 3. Surface Coverage Verteilung
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist([s*100 for s in all_surface], bins=10, color='#7bed9f', alpha=0.8, edgecolor='#55aa6f')
ax3.axvline(np.mean(all_surface)*100, color='#ff6b35', linewidth=2, linestyle='--',
            label=f'Mean: {np.mean(all_surface)*100:.1f}%')
ax3.set_xlabel('Surface Coverage (%)')
ax3.set_ylabel('Anzahl Modelle')
ax3.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax3, 'Keypoints auf Oberflaeche')

# 4. Chamfer Distance sortiert
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(sorted(all_chamfer), color=COLOR, linewidth=1.5)
ax4.fill_between(range(len(all_chamfer)), sorted(all_chamfer), alpha=0.2, color=COLOR)
ax4.set_xlabel('Modell (sortiert)')
ax4.set_ylabel('Chamfer Distance')
style_ax(ax4, 'Chamfer Distance (sortiert)')

# 5. Keypoint Positionen (alle Modelle überlagert, erste 2 Dimensionen)
ax5 = fig.add_subplot(gs[1, 1])
colors_kp = ['#ff0000','#ff8800','#ffff00','#00ff00','#00ffff','#0088ff','#ff00ff','#ffffff']
for k in range(ns.n_keypoint):
    ax5.scatter(kp_array[:, k, 0], kp_array[:, k, 1],
                c=colors_kp[k], s=2, alpha=0.3, label=f'KP{k+1}')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white',
           fontsize=7, markerscale=4, loc='upper right')
style_ax(ax5, 'Keypoint Positionen (alle Modelle)')

# 6. Zusammenfassung
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#16213e')
ax6.axis('off')

rows = [
    ("Metrik", "Wert"),
    ("─"*15, "─"*10),
    ("Testmodelle", f"{len(x_test)}"),
    ("Chamfer Mean", f"{m['chamfer_mean']:.4f}"),
    ("Chamfer Std", f"{m['chamfer_std']:.4f}"),
    ("KP Spread", f"{m['keypoint_spread_mean']:.4f}"),
    ("KP Konsistenz", f"{m['keypoint_consistency']:.4f}"),
    ("Surface Cover.", f"{m['surface_coverage_mean']*100:.1f}%"),
]

y = 0.95
for row in rows:
    ax6.text(0.05, y, row[0], transform=ax6.transAxes, color='#aaaaaa', fontsize=10, va='top')
    ax6.text(0.65, y, row[1], transform=ax6.transAxes, color=COLOR, fontsize=10, va='top', fontweight='bold')
    y -= 0.12

ax6.set_title('Zusammenfassung', color='white', fontsize=11, fontweight='bold', pad=10)
for spine in ax6.spines.values():
    spine.set_edgecolor('#444466')

fig.suptitle('Key-Grid Airplane V2 – Test Evaluation',
             color='white', fontsize=15, fontweight='bold', y=0.98)

plot_path = os.path.join(ns.output_dir, 'test_evaluation_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()

print(f"\n  Plot gespeichert:  {plot_path}")
print(f"  JSON gespeichert:  {json_path}")
print("\n" + "="*60)
print("  EVALUIERUNG ABGESCHLOSSEN")
print("="*60)
