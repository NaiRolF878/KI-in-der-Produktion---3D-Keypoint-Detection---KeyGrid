# -*- coding: utf-8 -*-
"""
Test-Evaluierung: Key-Grid Airplane V4 (k=12 Keypoints)
Mit Einzelauswertung V4 + Dreier-Vergleich: V2 (k=8) vs V3 (k=10) vs V4 (k=12)
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

arg_parser = argparse.ArgumentParser(description="Test-Evaluierung Key-Grid Airplane V4 k=12")
arg_parser.add_argument('--dataset-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\02691156')
arg_parser.add_argument('--keygrid-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid')
arg_parser.add_argument('--model', type=str,
                        default=r'C:\Users\xrstu\Key-Grid\model\airplane_v4_k12.pt')
arg_parser.add_argument('--n-keypoint', type=int, default=12)
arg_parser.add_argument('--max-points', type=int, default=2048)
arg_parser.add_argument('--batch', type=int, default=16)
arg_parser.add_argument('--output-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid\results\evaluation_airplane_v4_k12')
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
print("  Key-Grid Airplane V4 (k=12) – Test Evaluierung")
print("="*60)

np.random.seed(42)
x_all = load_pcd_files(ns.dataset_root, ns.max_points)
np.random.shuffle(x_all)
x_test = x_all[int(len(x_all) * 0.85):]
print(f"  Testdaten: {len(x_test)} Point Clouds")

# ── Modell laden ───────────────────────────────────────────────────────────
print(f"\n  Lade Modell: {ns.model}")
sys.path.insert(0, ns.keygrid_dir)
from merger.merger_net import Net

net = Net(ns.max_points, ns.n_keypoint).cuda()
net.load_state_dict(torch.load(ns.model, map_location='cuda', weights_only=False)['model_state_dict'])
net.eval()
print(f"  Modell geladen. ({ns.n_keypoint} Keypoints)")

# ── Metriken ───────────────────────────────────────────────────────────────
def chamfer_distance(pc1, pc2):
    diff = pc1[:, None, :] - pc2[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    return float(np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0)))

def keypoint_spread(keypoints):
    dists = []
    for i in range(len(keypoints)):
        for j in range(i+1, len(keypoints)):
            dists.append(np.linalg.norm(keypoints[i] - keypoints[j]))
    return float(np.mean(dists))

def keypoint_on_surface(keypoints, pointcloud, threshold=0.05):
    count = sum(1 for kp in keypoints if np.min(np.linalg.norm(pointcloud - kp, axis=1)) < threshold)
    return count / len(keypoints)

# ── Evaluierung ────────────────────────────────────────────────────────────
print("\n  Evaluiere auf Testdaten...")
all_keypoints, all_chamfer, all_spread, all_surface = [], [], [], []

with torch.no_grad():
    for i in range(0, len(x_test), ns.batch):
        batch_x = torch.FloatTensor(x_test[i:i+ns.batch]).cuda()
        keypoint, reconstruct = net(batch_x, 'False')
        for j in range(len(batch_x)):
            pc = batch_x[j].cpu().numpy()
            kp = keypoint[j].cpu().numpy()
            rc = reconstruct[j].cpu().numpy()
            all_keypoints.append(kp)
            all_chamfer.append(chamfer_distance(pc, rc))
            all_spread.append(keypoint_spread(kp))
            all_surface.append(keypoint_on_surface(kp, pc))
        print(f"  Processed {min(i+ns.batch, len(x_test))}/{len(x_test)}")

kp_array = np.array(all_keypoints)
consistency = float(np.mean(np.std(kp_array, axis=0)))

results = {
    'model': ns.model, 'category': 'Airplane', 'version': 'v4_k12',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'n_test': len(x_test), 'n_keypoints': ns.n_keypoint,
    'metrics': {
        'chamfer_mean':          float(np.mean(all_chamfer)),
        'chamfer_std':           float(np.std(all_chamfer)),
        'chamfer_min':           float(np.min(all_chamfer)),
        'chamfer_max':           float(np.max(all_chamfer)),
        'keypoint_spread_mean':  float(np.mean(all_spread)),
        'keypoint_spread_std':   float(np.std(all_spread)),
        'keypoint_consistency':  consistency,
        'surface_coverage_mean': float(np.mean(all_surface)),
    }
}

json_path = os.path.join(ns.output_dir, 'test_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

m = results['metrics']
print("\n" + "="*60)
print("  TESTERGEBNISSE – Airplane V4 (k=12)")
print("="*60)
print(f"  Testmodelle:           {len(x_test)}")
print(f"  Keypoints:             {ns.n_keypoint}")
print(f"  Chamfer Distance:      {m['chamfer_mean']:.4f} ± {m['chamfer_std']:.4f}")
print(f"  Chamfer Min/Max:       {m['chamfer_min']:.4f} / {m['chamfer_max']:.4f}")
print(f"  Keypoint Spread:       {m['keypoint_spread_mean']:.4f} ± {m['keypoint_spread_std']:.4f}")
print(f"  Keypoint Konsistenz:   {m['keypoint_consistency']:.4f}  (niedriger = besser)")
print(f"  Surface Coverage:      {m['surface_coverage_mean']*100:.1f}%")
print("="*60)

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 1 – Einzelauswertung V4
# ══════════════════════════════════════════════════════════════════════════
COLOR = '#7bed9f'

fig1 = plt.figure(figsize=(16, 10))
fig1.patch.set_facecolor('#1a1a2e')
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.4, wspace=0.35)

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
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.hist(all_chamfer, bins=20, color=COLOR, alpha=0.8, edgecolor='#55aa6f')
ax1.axvline(np.mean(all_chamfer), color='#ff6b35', linewidth=2, linestyle='--',
            label=f'Mean: {np.mean(all_chamfer):.4f}')
ax1.set_xlabel('Chamfer Distance')
ax1.set_ylabel('Anzahl Modelle')
ax1.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax1, 'Chamfer Distance Verteilung')

# 2. Keypoint Spread Verteilung
ax2 = fig1.add_subplot(gs1[0, 1])
ax2.hist(all_spread, bins=20, color='#ff6b35', alpha=0.8, edgecolor='#cc4400')
ax2.axvline(np.mean(all_spread), color=COLOR, linewidth=2, linestyle='--',
            label=f'Mean: {np.mean(all_spread):.4f}')
ax2.set_xlabel('Keypoint Spread')
ax2.set_ylabel('Anzahl Modelle')
ax2.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax2, 'Keypoint Spread Verteilung')

# 3. Surface Coverage
ax3 = fig1.add_subplot(gs1[0, 2])
ax3.hist([s*100 for s in all_surface], bins=10, color='#00d4ff', alpha=0.8, edgecolor='#0099bb')
ax3.axvline(np.mean(all_surface)*100, color='#ff6b35', linewidth=2, linestyle='--',
            label=f'Mean: {np.mean(all_surface)*100:.1f}%')
ax3.set_xlabel('Surface Coverage (%)')
ax3.set_ylabel('Anzahl Modelle')
ax3.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax3, 'Keypoints auf Oberflaeche')

# 4. Chamfer Distance sortiert
ax4 = fig1.add_subplot(gs1[1, 0])
ax4.plot(sorted(all_chamfer), color=COLOR, linewidth=1.5)
ax4.fill_between(range(len(all_chamfer)), sorted(all_chamfer), alpha=0.2, color=COLOR)
ax4.set_xlabel('Modell (sortiert)')
ax4.set_ylabel('Chamfer Distance')
style_ax(ax4, 'Chamfer Distance (sortiert)')

# 5. Keypoint Positionen
ax5 = fig1.add_subplot(gs1[1, 1])
colors_kp = ['#ff0000','#ff8800','#ffff00','#00ff00','#00ffff','#0088ff',
             '#ff00ff','#ffffff','#ff6666','#66ff66','#6666ff','#ffaa00']
for k in range(ns.n_keypoint):
    ax5.scatter(kp_array[:, k, 0], kp_array[:, k, 1],
                c=colors_kp[k], s=2, alpha=0.3, label=f'KP{k+1}')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white',
           fontsize=6, markerscale=4, loc='upper right', ncol=2)
style_ax(ax5, 'Keypoint Positionen (alle Modelle)')

# 6. Zusammenfassung
ax6 = fig1.add_subplot(gs1[1, 2])
ax6.set_facecolor('#16213e')
ax6.axis('off')

rows_summary = [
    ("Metrik", "Wert"),
    ("─"*15, "─"*10),
    ("Testmodelle",    f"{len(x_test)}"),
    ("Keypoints",      f"{ns.n_keypoint}"),
    ("Chamfer Mean",   f"{m['chamfer_mean']:.4f}"),
    ("Chamfer Std",    f"{m['chamfer_std']:.4f}"),
    ("Chamfer Min",    f"{m['chamfer_min']:.4f}"),
    ("Chamfer Max",    f"{m['chamfer_max']:.4f}"),
    ("KP Spread",      f"{m['keypoint_spread_mean']:.4f}"),
    ("KP Konsistenz",  f"{m['keypoint_consistency']:.4f}"),
    ("Surface Cover.", f"{m['surface_coverage_mean']*100:.1f}%"),
]

y = 0.97
for row in rows_summary:
    ax6.text(0.05, y, row[0], transform=ax6.transAxes, color='#aaaaaa', fontsize=9, va='top')
    ax6.text(0.62, y, row[1], transform=ax6.transAxes, color=COLOR, fontsize=9, va='top', fontweight='bold')
    y -= 0.09

ax6.set_title('Zusammenfassung V4 (k=12)', color='white', fontsize=11, fontweight='bold', pad=10)
for spine in ax6.spines.values():
    spine.set_edgecolor('#444466')

fig1.suptitle('Key-Grid Airplane V4 (k=12) – Einzelauswertung',
              color='white', fontsize=15, fontweight='bold', y=0.98)

plot1_path = os.path.join(ns.output_dir, 'v4_einzelauswertung_plot.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"\n  Plot 1 gespeichert: {plot1_path}")

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 2 – Dreier-Vergleich k=8 / k=10 / k=12
# ══════════════════════════════════════════════════════════════════════════
v2_json = r'C:\Users\xrstu\Key-Grid\results\evaluation_airplane_v2\test_results.json'
v3_json = r'C:\Users\xrstu\Key-Grid\results\evaluation_airplane_v3_k10\test_results.json'

versions = {}
if os.path.exists(v2_json):
    with open(v2_json) as f:
        versions['V2 (k=8)'] = json.load(f)['metrics']
if os.path.exists(v3_json):
    with open(v3_json) as f:
        versions['V3 (k=10)'] = json.load(f)['metrics']
versions['V4 (k=12)'] = m

COLORS_V = {'V2 (k=8)': '#00d4ff', 'V3 (k=10)': '#ff6b35', 'V4 (k=12)': '#7bed9f'}

fig2 = plt.figure(figsize=(18, 12))
fig2.patch.set_facecolor('#1a1a2e')
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.4, wspace=0.35)

vnames = list(versions.keys())

# 1. Chamfer Distance
ax1 = fig2.add_subplot(gs2[0, 0])
chamfer_vals = [versions[v]['chamfer_mean'] for v in vnames]
chamfer_stds = [versions[v]['chamfer_std'] for v in vnames]
bars = ax1.bar(vnames, chamfer_vals, yerr=chamfer_stds,
               color=[COLORS_V[v] for v in vnames], alpha=0.8, capsize=5)
ax1.set_ylabel('Chamfer Distance')
for bar, val in zip(bars, chamfer_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9)
style_ax(ax1, 'Chamfer Distance Vergleich')

# 2. Keypoint Spread
ax2 = fig2.add_subplot(gs2[0, 1])
spread_vals = [versions[v]['keypoint_spread_mean'] for v in vnames]
spread_stds = [versions[v]['keypoint_spread_std'] for v in vnames]
bars2 = ax2.bar(vnames, spread_vals, yerr=spread_stds,
                color=[COLORS_V[v] for v in vnames], alpha=0.8, capsize=5)
ax2.set_ylabel('Keypoint Spread')
for bar, val in zip(bars2, spread_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9)
style_ax(ax2, 'Keypoint Spread Vergleich')

# 3. Konsistenz
ax3 = fig2.add_subplot(gs2[0, 2])
consist_vals = [versions[v]['keypoint_consistency'] for v in vnames]
bars3 = ax3.bar(vnames, consist_vals,
                color=[COLORS_V[v] for v in vnames], alpha=0.8)
ax3.set_ylabel('Konsistenz (niedriger = besser)')
for bar, val in zip(bars3, consist_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9)
style_ax(ax3, 'Keypoint Konsistenz Vergleich')

# 4. Surface Coverage
ax4 = fig2.add_subplot(gs2[1, 0])
surface_vals = [versions[v]['surface_coverage_mean']*100 for v in vnames]
bars4 = ax4.bar(vnames, surface_vals,
                color=[COLORS_V[v] for v in vnames], alpha=0.8)
ax4.set_ylabel('Surface Coverage (%)')
for bar, val in zip(bars4, surface_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=9)
style_ax(ax4, 'Surface Coverage Vergleich')

# 5. Chamfer Std
ax5 = fig2.add_subplot(gs2[1, 1])
chamfer_std_vals = [versions[v]['chamfer_std'] for v in vnames]
bars5 = ax5.bar(vnames, chamfer_std_vals,
                color=[COLORS_V[v] for v in vnames], alpha=0.8)
ax5.set_ylabel('Chamfer Std')
for bar, val in zip(bars5, chamfer_std_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9)
style_ax(ax5, 'Chamfer Std Vergleich')

# 6. Zusammenfassungstabelle
ax6 = fig2.add_subplot(gs2[1, 2])
ax6.set_facecolor('#16213e')
ax6.axis('off')

header = ["Metrik"] + vnames
rows_data = [
    ["Chamfer Mean"] + [f"{versions[v]['chamfer_mean']:.4f}" for v in vnames],
    ["Chamfer Std"]  + [f"{versions[v]['chamfer_std']:.4f}"  for v in vnames],
    ["KP Spread"]    + [f"{versions[v]['keypoint_spread_mean']:.4f}" for v in vnames],
    ["Konsistenz"]   + [f"{versions[v]['keypoint_consistency']:.4f}" for v in vnames],
    ["Surface Cover"]+ [f"{versions[v]['surface_coverage_mean']*100:.1f}%" for v in vnames],
]

y = 0.95
ax6.text(0.02, y, header[0], transform=ax6.transAxes, color='#aaaaaa', fontsize=8, va='top', fontweight='bold')
for ci, (vname, col) in enumerate(zip(vnames, [COLORS_V[v] for v in vnames])):
    ax6.text(0.35 + ci*0.22, y, vname, transform=ax6.transAxes, color=col, fontsize=8, va='top', fontweight='bold')
y -= 0.08
ax6.text(0.02, y, "─"*40, transform=ax6.transAxes, color='#444466', fontsize=7, va='top')
y -= 0.08

for row in rows_data:
    ax6.text(0.02, y, row[0], transform=ax6.transAxes, color='#aaaaaa', fontsize=8, va='top')
    for ci, (val, vname) in enumerate(zip(row[1:], vnames)):
        ax6.text(0.35 + ci*0.22, y, val, transform=ax6.transAxes,
                 color=COLORS_V[vname], fontsize=8, va='top', fontweight='bold')
    y -= 0.10

ax6.set_title('Zusammenfassung k=8 / k=10 / k=12', color='white', fontsize=11, fontweight='bold', pad=10)
for spine in ax6.spines.values():
    spine.set_edgecolor('#444466')

fig2.suptitle('Key-Grid Airplane – Vergleich k=8 vs k=10 vs k=12',
              color='white', fontsize=15, fontweight='bold', y=1.02)

plot2_path = os.path.join(ns.output_dir, 'comparison_k8_k10_k12_plot.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"  Plot 2 gespeichert: {plot2_path}")

print(f"  JSON gespeichert:   {json_path}")
print("\n" + "="*60)
print("  EVALUIERUNG ABGESCHLOSSEN")
print("="*60)