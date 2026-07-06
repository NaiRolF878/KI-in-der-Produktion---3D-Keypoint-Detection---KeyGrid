# -*- coding: utf-8 -*-
"""
Keypoint Prediction in Annotationsreihenfolge
Lädt die .pcd Dateien in der Reihenfolge der KeypointNet Annotationen
Speichert Keypoints mit model_id für PCK Vergleich
"""
import torch
import merger.merger_net as merger_net
import numpy as np
import argparse
import open3d as o3d
import os
import json

arg_parser = argparse.ArgumentParser(description="Predictor in Annotationsreihenfolge")
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str,
                        default='model/airplane_v3_k10.pt')
arg_parser.add_argument('-d', '--device', type=str, default='cuda')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10)
arg_parser.add_argument('--max-points', type=int, default=2048)
arg_parser.add_argument('--annotation-json', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\annotations\airplane.json',
                        help='Pfad zur KeypointNet Annotation JSON')
arg_parser.add_argument('--pcd-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds',
                        help='Root Ordner der pcd Dateien')
arg_parser.add_argument('--output', type=str,
                        default='results/airplane_v3_k10_annotated.npz',
                        help='Ausgabe .npz mit Keypoints und model_ids')
arg_parser.add_argument('--batch', type=int, default=16)

ns = arg_parser.parse_args()

os.makedirs(os.path.dirname(ns.output) if os.path.dirname(ns.output) else '.', exist_ok=True)

# ── Annotationen laden ─────────────────────────────────────────────────────
print(f"Lade Annotationen: {ns.annotation_json}")
annotations = json.load(open(ns.annotation_json))
print(f"  {len(annotations)} annotierte Modelle gefunden")

# ── Modell laden ───────────────────────────────────────────────────────────
print(f"Lade Modell: {ns.checkpoint_path}")
net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path,
                               map_location=torch.device(ns.device),
                               weights_only=False)['model_state_dict'])
net.eval()
print(f"  Modell geladen. (k={ns.n_keypoint})")

# ── Point Clouds in Annotationsreihenfolge laden ───────────────────────────
def load_pcd(path, max_points=2048):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        return None
    if len(pts) >= max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
    else:
        idx = np.random.choice(len(pts), max_points, replace=True)
    return pts[idx]

print("\nLade Point Clouds in Annotationsreihenfolge...")
valid_entries = []
point_clouds  = []
skipped       = 0

for entry in annotations:
    cid = entry['class_id']
    mid = entry['model_id']
    pcd_path = os.path.join(ns.pcd_root, cid, f"{mid}.pcd")

    if not os.path.exists(pcd_path):
        skipped += 1
        continue

    pts = load_pcd(pcd_path, ns.max_points)
    if pts is None:
        skipped += 1
        continue

    valid_entries.append(entry)
    point_clouds.append(pts)

print(f"  Geladen: {len(point_clouds)} | Übersprungen: {skipped}")

# ── Keypoints vorhersagen ──────────────────────────────────────────────────
print("\nVorhersage Keypoints...")
x = np.array(point_clouds, dtype=np.float32)
all_keypoints = []

with torch.no_grad():
    for i in range(0, len(x), ns.batch):
        batch = torch.FloatTensor(x[i:i+ns.batch]).to(ns.device)
        key_points, _ = net(batch, 'False')
        for kp in key_points:
            all_keypoints.append(kp.cpu().numpy())
        print(f"  Processed {min(i+ns.batch, len(x))}/{len(x)}")

keypoints_array = np.array(all_keypoints)  # (N, k, 3)
model_ids = [e['model_id'] for e in valid_entries]
class_ids = [e['class_id'] for e in valid_entries]

# Ground Truth Keypoints auch speichern für direkten Vergleich
gt_keypoints = []
gt_semantic_ids = []
for entry in valid_entries:
    kps = entry['keypoints']
    gt_keypoints.append([kp['xyz'] for kp in kps])
    gt_semantic_ids.append([kp['semantic_id'] for kp in kps])

# ── Speichern ──────────────────────────────────────────────────────────────
np.savez(
    ns.output,
    keypoints=keypoints_array,           # (N, k, 3) predicted
    model_ids=np.array(model_ids),       # (N,) model_id strings
    class_ids=np.array(class_ids),       # (N,) class_id strings
    point_clouds=x,                      # (N, 2048, 3) point clouds
)

# Ground Truth separat als JSON (wegen variabler Länge pro Modell)
gt_path = ns.output.replace('.npz', '_gt.json')
with open(gt_path, 'w') as f:
    json.dump({
        'model_ids': model_ids,
        'gt_keypoints': gt_keypoints,
        'gt_semantic_ids': gt_semantic_ids,
    }, f, indent=2)

print(f"\nGespeichert:")
print(f"  Predicted Keypoints: {ns.output}")
print(f"  Ground Truth:        {gt_path}")
print(f"  Shape:               {keypoints_array.shape}  (modelle x keypoints x 3)")
print(f"  Modelle:             {len(model_ids)}")
