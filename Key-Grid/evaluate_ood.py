# -*- coding: utf-8 -*-
"""
Out-of-Distribution (OoD) Test
Zeigt die Schwäche des Modells: trainiert auf Airplane, getestet auf Mug
Vergleicht In-Distribution (Airplane) vs Out-of-Distribution (Mug)
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

arg_parser = argparse.ArgumentParser(description="OoD Test: Airplane Modell auf fremden Daten")
arg_parser.add_argument('--keygrid-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid')
arg_parser.add_argument('--model', type=str,
                        default=r'C:\Users\xrstu\Key-Grid\model\airplane_v2.pt',
                        help='Trainiertes Airplane Modell')
arg_parser.add_argument('--in-dist-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\02691156',
                        help='In-Distribution: Airplane')
arg_parser.add_argument('--ood-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\03797390',
                        help='Out-of-Distribution: Mug')
arg_parser.add_argument('--n-keypoint', type=int, default=8)
arg_parser.add_argument('--max-points', type=int, default=2048)
arg_parser.add_argument('--batch', type=int, default=16)
arg_parser.add_argument('--n-samples', type=int, default=100,
                        help='Anzahl Modelle pro Kategorie zum Testen')
arg_parser.add_argument('--output-dir', type=str,
                        default=r'C:\Users\xrstu\Key-Grid\results\ood_evaluation')
ns = arg_parser.parse_args()

os.makedirs(ns.output_dir, exist_ok=True)

# ── Daten laden ────────────────────────────────────────────────────────────
def load_pcd_files(folder, max_points=2048, n_samples=None):
    point_clouds = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
    if n_samples:
        files = files[:n_samples]
    print(f"  Lade {len(files)} .pcd Dateien aus {os.path.basename(folder)}")
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

def keypoint_consistency(kp_list):
    kp_array = np.array(kp_list)
    return float(np.mean(np.std(kp_array, axis=0)))

# ── Modell laden ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  OoD Test: Airplane Modell vs fremde Daten")
print("="*60)
print(f"\n  Lade Modell: {ns.model}")
sys.path.insert(0, ns.keygrid_dir)
from merger.merger_net import Net

net = Net(ns.max_points, ns.n_keypoint).cuda()
net.load_state_dict(torch.load(ns.model, map_location='cuda', weights_only=False)['model_state_dict'])
net.eval()
print(f"  Modell geladen. (trainiert auf Airplane, k={ns.n_keypoint})")

# ── Evaluierungsfunktion ───────────────────────────────────────────────────
def evaluate(x_data, label):
    print(f"\n  Evaluiere: {label} ({len(x_data)} Modelle)...")
    all_kp, all_cd, all_spread, all_surface = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(x_data), ns.batch):
            batch_x = torch.FloatTensor(x_data[i:i+ns.batch]).cuda()
            keypoint, reconstruct = net(batch_x, 'False')
            for j in range(len(batch_x)):
                pc = batch_x[j].cpu().numpy()
                kp = keypoint[j].cpu().numpy()
                rc = reconstruct[j].cpu().numpy()
                all_kp.append(kp)
                all_cd.append(chamfer_distance(pc, rc))
                all_spread.append(keypoint_spread(kp))
                all_surface.append(keypoint_on_surface(kp, pc))

    return {
        'label': label,
        'n': len(x_data),
        'chamfer_mean':    float(np.mean(all_cd)),
        'chamfer_std':     float(np.std(all_cd)),
        'spread_mean':     float(np.mean(all_spread)),
        'spread_std':      float(np.std(all_spread)),
        'consistency':     keypoint_consistency(all_kp),
        'surface_mean':    float(np.mean(all_surface)),
        'all_cd':          all_cd,
        'all_spread':      all_spread,
        'all_surface':     all_surface,
        'all_kp':          [kp.tolist() for kp in all_kp],
    }

# ── Daten laden und evaluieren ─────────────────────────────────────────────
x_in  = load_pcd_files(ns.in_dist_root, ns.max_points, ns.n_samples)
x_ood = load_pcd_files(ns.ood_root,     ns.max_points, ns.n_samples)

res_in  = evaluate(x_in,  'Airplane (In-Distribution)')
res_ood = evaluate(x_ood, 'Mug (Out-of-Distribution)')

# ── Ergebnisse ausgeben ────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ERGEBNISSE")
print("="*60)
print(f"  {'Metrik':<25} {'Airplane (In)':>15} {'Mug (OoD)':>15} {'Faktor':>8}")
print(f"  {'-'*65}")

def factor(in_val, ood_val):
    if in_val > 0:
        return f"{ood_val/in_val:.1f}x"
    return "N/A"

rows = [
    ("Chamfer Distance",  res_in['chamfer_mean'],  res_ood['chamfer_mean']),
    ("Chamfer Std",       res_in['chamfer_std'],   res_ood['chamfer_std']),
    ("Keypoint Spread",   res_in['spread_mean'],   res_ood['spread_mean']),
    ("Konsistenz",        res_in['consistency'],   res_ood['consistency']),
    ("Surface Coverage",  res_in['surface_mean'],  res_ood['surface_mean']),
]
for name, in_val, ood_val in rows:
    print(f"  {name:<25} {in_val:>15.4f} {ood_val:>15.4f} {factor(in_val, ood_val):>8}")

print("="*60)
print(f"\n  Interpretation:")
print(f"  → Chamfer {factor(res_in['chamfer_mean'], res_ood['chamfer_mean'])} höher bei OoD = Rekonstruktion schlechter")
print(f"  → Konsistenz {factor(res_in['consistency'], res_ood['consistency'])} höher bei OoD = Keypoints inkonsistenter")
print(f"  → Surface Coverage {res_ood['surface_mean']*100:.1f}% bei OoD = Keypoints teilweise daneben")

# JSON speichern
json_path = os.path.join(ns.output_dir, 'ood_results.json')
save_data = {
    'model': ns.model,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'in_distribution': {k: v for k, v in res_in.items() if k not in ['all_cd', 'all_spread', 'all_surface', 'all_kp']},
    'out_of_distribution': {k: v for k, v in res_ood.items() if k not in ['all_cd', 'all_spread', 'all_surface', 'all_kp']},
}
with open(json_path, 'w') as f:
    json.dump(save_data, f, indent=2)

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

COLOR_IN  = '#00d4ff'   # Blau = In-Distribution
COLOR_OOD = '#ff6b35'   # Orange = Out-of-Distribution

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
ax1.hist(res_in['all_cd'],  bins=20, color=COLOR_IN,  alpha=0.7, label='Airplane (In)', edgecolor='#0099bb')
ax1.hist(res_ood['all_cd'], bins=20, color=COLOR_OOD, alpha=0.7, label='Mug (OoD)',     edgecolor='#cc4400')
ax1.axvline(res_in['chamfer_mean'],  color=COLOR_IN,  linewidth=2, linestyle='--')
ax1.axvline(res_ood['chamfer_mean'], color=COLOR_OOD, linewidth=2, linestyle='--')
ax1.set_xlabel('Chamfer Distance')
ax1.set_ylabel('Anzahl Modelle')
ax1.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax1, 'Chamfer Distance: In vs OoD')

# 2. Keypoint Spread Verteilung
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(res_in['all_spread'],  bins=20, color=COLOR_IN,  alpha=0.7, label='Airplane (In)')
ax2.hist(res_ood['all_spread'], bins=20, color=COLOR_OOD, alpha=0.7, label='Mug (OoD)')
ax2.axvline(res_in['spread_mean'],  color=COLOR_IN,  linewidth=2, linestyle='--')
ax2.axvline(res_ood['spread_mean'], color=COLOR_OOD, linewidth=2, linestyle='--')
ax2.set_xlabel('Keypoint Spread')
ax2.set_ylabel('Anzahl Modelle')
ax2.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax2, 'Keypoint Spread: In vs OoD')

# 3. Surface Coverage Vergleich
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist([s*100 for s in res_in['all_surface']],  bins=10, color=COLOR_IN,  alpha=0.7, label='Airplane (In)')
ax3.hist([s*100 for s in res_ood['all_surface']], bins=10, color=COLOR_OOD, alpha=0.7, label='Mug (OoD)')
ax3.set_xlabel('Surface Coverage (%)')
ax3.set_ylabel('Anzahl Modelle')
ax3.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax3, 'Surface Coverage: In vs OoD')

# 4. Balkenvergleich Metriken
ax4 = fig.add_subplot(gs[1, 0])
metriken = ['Chamfer\nMean', 'Chamfer\nStd', 'KP\nSpread', 'Konsistenz']
in_vals  = [res_in['chamfer_mean'],  res_in['chamfer_std'],  res_in['spread_mean'],  res_in['consistency']]
ood_vals = [res_ood['chamfer_mean'], res_ood['chamfer_std'], res_ood['spread_mean'], res_ood['consistency']]
x = np.arange(len(metriken))
w = 0.35
ax4.bar(x - w/2, in_vals,  w, color=COLOR_IN,  alpha=0.8, label='Airplane (In)')
ax4.bar(x + w/2, ood_vals, w, color=COLOR_OOD, alpha=0.8, label='Mug (OoD)')
ax4.set_xticks(x)
ax4.set_xticklabels(metriken, color='#aaaaaa', fontsize=9)
ax4.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax4, 'Metriken Vergleich')

# 5. Chamfer Distance sortiert (beide)
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(sorted(res_in['all_cd']),  color=COLOR_IN,  linewidth=2, label='Airplane (In)')
ax5.plot(sorted(res_ood['all_cd']), color=COLOR_OOD, linewidth=2, label='Mug (OoD)')
ax5.fill_between(range(len(res_in['all_cd'])),  sorted(res_in['all_cd']),  alpha=0.15, color=COLOR_IN)
ax5.fill_between(range(len(res_ood['all_cd'])), sorted(res_ood['all_cd']), alpha=0.15, color=COLOR_OOD)
ax5.set_xlabel('Modell (sortiert)')
ax5.set_ylabel('Chamfer Distance')
ax5.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
style_ax(ax5, 'Chamfer Distance (sortiert)')

# 6. Zusammenfassung
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#16213e')
ax6.axis('off')

summary_rows = [
    ("Metrik", "Airplane", "Mug (OoD)", "Faktor"),
    ("─"*8, "─"*8, "─"*8, "─"*6),
    ("Chamfer Mean",
     f"{res_in['chamfer_mean']:.4f}",
     f"{res_ood['chamfer_mean']:.4f}",
     factor(res_in['chamfer_mean'], res_ood['chamfer_mean'])),
    ("Chamfer Std",
     f"{res_in['chamfer_std']:.4f}",
     f"{res_ood['chamfer_std']:.4f}",
     factor(res_in['chamfer_std'], res_ood['chamfer_std'])),
    ("KP Spread",
     f"{res_in['spread_mean']:.4f}",
     f"{res_ood['spread_mean']:.4f}",
     factor(res_in['spread_mean'], res_ood['spread_mean'])),
    ("Konsistenz",
     f"{res_in['consistency']:.4f}",
     f"{res_ood['consistency']:.4f}",
     factor(res_in['consistency'], res_ood['consistency'])),
    ("Surface Cover.",
     f"{res_in['surface_mean']*100:.1f}%",
     f"{res_ood['surface_mean']*100:.1f}%",
     ""),
]

y = 0.97
for row in summary_rows:
    ax6.text(0.01, y, row[0], transform=ax6.transAxes, color='#aaaaaa', fontsize=8, va='top')
    ax6.text(0.35, y, row[1], transform=ax6.transAxes, color=COLOR_IN,  fontsize=8, va='top', fontweight='bold')
    ax6.text(0.58, y, row[2], transform=ax6.transAxes, color=COLOR_OOD, fontsize=8, va='top', fontweight='bold')
    ax6.text(0.83, y, row[3], transform=ax6.transAxes, color='#ffff00', fontsize=8, va='top', fontweight='bold')
    y -= 0.12

ax6.set_title('OoD Zusammenfassung', color='white', fontsize=11, fontweight='bold', pad=10)
for spine in ax6.spines.values():
    spine.set_edgecolor('#444466')

fig.suptitle('Out-of-Distribution Test: Airplane Modell auf Mug Daten',
             color='white', fontsize=14, fontweight='bold', y=0.98)

plot_path = os.path.join(ns.output_dir, 'ood_evaluation_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()

print(f"\n  Plot gespeichert:  {plot_path}")
print(f"  JSON gespeichert:  {json_path}")
print("\n" + "="*60)
print("  OoD EVALUIERUNG ABGESCHLOSSEN")
print("="*60)
