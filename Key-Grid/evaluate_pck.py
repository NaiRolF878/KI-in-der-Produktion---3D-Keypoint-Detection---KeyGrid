# -*- coding: utf-8 -*-
"""
evaluate_pck.py — PCK-Auswertung Key-Grid Airplane gegen KeypointNet Ground Truth.

ALL-IN-ONE: laedt PCDs, sagt Keypoints vorher, rekonstruiert GT-Keypoints aus
der Annotation und berechnet PCK — alles in einem Lauf, damit keine
Reihenfolge- oder Koordinaten-Diskrepanzen entstehen koennen.

WARUM SO:
  - Die GT-Keypoints werden NICHT ueber ihre rohen xyz genommen, sondern ueber
    pcd_info.point_index direkt aus DERSELBEN Punktwolke gelesen, die auch das
    Modell sieht. Dadurch liegen GT und Prediction garantiert im selben Raum,
    egal wie KeypointNet die Koordinaten normalisiert hat.
  - GT und PCD werden ueber den Dateinamen (= model_id) verknuepft, nicht ueber
    Array-Indizes. Kein fragiles Alignment zwischen getrennten Laeufen.

WICHTIG zur Punkt-Auswahl:
  Das Subsampling im Lade-Schritt nutzt KEIN np.random.choice. Damit point_index
  gueltig bleibt, werden die ERSTEN max_points genommen bzw. die volle Wolke
  verwendet. point_index bezieht sich auf die ORIGINAL-Punktreihenfolge der .pcd.

Beispiel (Anaconda Prompt, env keygrid):
  python evaluate_pck.py ^
    --gt-json   C:\\Users\\xrstu\\KeypointNet\\annotations\\airplane.json ^
    --dataset-root C:\\Users\\xrstu\\KeypointNet\\pcds\\02691156 ^
    --checkpoint-path model\\airplane_v2.pt ^
    --n-keypoint 8 ^
    --output    pck_airplane_v2.json

Optional, nur auf dem Testsplit auswerten:
    --test-split-only
"""

import os
import json
import argparse
import numpy as np

import torch
import open3d as o3d
from scipy.optimize import linear_sum_assignment

import merger.merger_net as merger_net


# ----------------------------------------------------------------------------
# Argumente
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="PCK-Auswertung Key-Grid vs. KeypointNet GT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gt-json', required=True,
                   help='annotations/<kategorie>.json (KeypointNet)')
    p.add_argument('--dataset-root', type=str,
                   default=r'C:\Users\xrstu\KeypointNet\pcds\02691156',
                   help='Ordner mit den .pcd Dateien')
    p.add_argument('-m', '--checkpoint-path', type=str, default='model/airplane_v2.pt')
    p.add_argument('-k', '--n-keypoint', type=int, default=8)
    p.add_argument('--max-points', type=int, default=2048)
    p.add_argument('-d', '--device', type=str, default='cuda')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--output', type=str, default='pck_results.json')
    p.add_argument('--thresholds', type=str,
                   default='0.01,0.02,0.05,0.08,0.1,0.15,0.2',
                   help='Schwellwerte als Anteil der BBox-Diagonale')
    p.add_argument('--test-split-only', action='store_true',
                   help='Nur den 15-Prozent-Testsplit auswerten (Seed 42)')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Ground Truth
# ----------------------------------------------------------------------------
def load_ground_truth(gt_json_path):
    """model_id -> {'point_index': [...], 'sem': (K,)}"""
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    gt = {}
    for entry in data:
        kps = entry['keypoints']
        gt[entry['model_id']] = {
            'point_index': [kp['pcd_info']['point_index'] for kp in kps],
            'sem': np.array([kp['semantic_id'] for kp in kps], dtype=np.int64),
        }
    return gt


# ----------------------------------------------------------------------------
# PCDs laden — OHNE zufaelliges Resampling, damit point_index gueltig bleibt
# ----------------------------------------------------------------------------
def load_pcd_indexed(folder, gt, max_points):
    """
    Laedt nur die .pcd Dateien, fuer die eine GT existiert.
    Rueckgabe:
      model_ids   : Liste der geladenen model_ids
      clouds_net  : (N, max_points, 3) float32  -> Input fuers Netz
      gt_xyz_list : Liste (K,3) GT-Keypoints aus point_index der Originalwolke
    """
    model_ids, clouds_net, gt_xyz_list = [], [], []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
    missing_gt, bad_idx = 0, 0

    for fname in files:
        mid = os.path.splitext(fname)[0]
        if mid not in gt:
            missing_gt += 1
            continue

        pcd = o3d.io.read_point_cloud(os.path.join(folder, fname))
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            continue

        idxs = gt[mid]['point_index']
        if max(idxs) >= len(pts):
            bad_idx += 1
            continue
        gt_xyz = pts[idxs].astype(np.float64)          # garantiert gleicher Raum

        if len(pts) >= max_points:
            net_pts = pts[:max_points]
        else:
            reps = int(np.ceil(max_points / len(pts)))
            net_pts = np.tile(pts, (reps, 1))[:max_points]

        model_ids.append(mid)
        clouds_net.append(net_pts.astype(np.float32))
        gt_xyz_list.append(gt_xyz)

    print(f"  Geladen: {len(model_ids)} Modelle "
          f"(uebersprungen: {missing_gt} ohne GT, {bad_idx} mit ungueltigem point_index)")
    return model_ids, np.array(clouds_net, dtype=np.float32), gt_xyz_list


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict_keypoints(net, clouds, device, batch):
    preds = []
    with torch.no_grad():
        for i in range(0, len(clouds), batch):
            b = torch.FloatTensor(clouds[i:i+batch]).to(device)
            kp, _ = net(b, 'False')
            for k in kp:
                preds.append(k.cpu().numpy().astype(np.float64))
            print(f"  Predicted {min(i+batch, len(clouds))}/{len(clouds)}")
    return preds


# ----------------------------------------------------------------------------
# Geometrie / PCK
# ----------------------------------------------------------------------------
def bbox_diagonal(points):
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def matched_distances(gt_xyz, pred_xyz):
    cost = np.linalg.norm(gt_xyz[:, None, :] - pred_xyz[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    return cost[row, col]


def compute_pck(model_ids, gt_xyz_list, pred_list, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    correct = np.zeros_like(thresholds)
    total = 0
    per_model = {}

    for mid, gt_xyz, pred_xyz in zip(model_ids, gt_xyz_list, pred_list):
        if gt_xyz.size == 0 or pred_xyz.size == 0:
            continue
        scale = bbox_diagonal(gt_xyz) or 1.0
        d = matched_distances(gt_xyz, pred_xyz) / scale
        total += d.shape[0]
        correct += (d[None, :] <= thresholds[:, None]).sum(axis=1)
        mid_idx = len(thresholds) // 2
        per_model[mid] = {
            'n_keypoints': int(d.shape[0]),
            'mean_norm_dist': float(d.mean()),
            'pck_at_mid': float((d <= thresholds[mid_idx]).mean()),
        }

    pck = correct / max(total, 1)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    span = thresholds[-1] - thresholds[0]
    auc = float(trapz(pck, thresholds) / span) if span > 0 else float(pck[0])

    return {
        'thresholds': thresholds.tolist(),
        'pck_curve': pck.tolist(),
        'auc': auc,
        'n_models': len(per_model),
        'total_keypoints': total,
        'per_model': per_model,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ns = parse_args()
    thresholds = [float(t) for t in ns.thresholds.split(',')]

    print("Lade Ground Truth ...")
    gt = load_ground_truth(ns.gt_json)
    print(f"  {len(gt)} GT-Modelle.")

    print(f"Lade Point Clouds aus: {ns.dataset_root}")
    model_ids, clouds, gt_xyz_list = load_pcd_indexed(ns.dataset_root, gt, ns.max_points)

    if ns.test_split_only:
        order = np.arange(len(model_ids))
        np.random.RandomState(42).shuffle(order)
        start = int(len(order) * 0.85)
        sel = set(order[start:].tolist())
        keep = [i for i in range(len(model_ids)) if i in sel]
        model_ids   = [model_ids[i] for i in keep]
        clouds      = clouds[keep]
        gt_xyz_list = [gt_xyz_list[i] for i in keep]
        print(f"  Testsplit: {len(model_ids)} Modelle")

    print(f"Lade Modell: {ns.checkpoint_path}")
    net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
    ckpt = torch.load(ns.checkpoint_path, map_location=torch.device(ns.device),
                      weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    print("Sage Keypoints vorher ...")
    preds = predict_keypoints(net, clouds, ns.device, ns.batch)

    print("Berechne PCK ...")
    results = compute_pck(model_ids, gt_xyz_list, preds, thresholds)

    with open(ns.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 30)
    print(f"  Modelle ausgewertet : {results['n_models']}")
    print(f"  Keypoints gesamt    : {results['total_keypoints']}")
    print(f"{'Threshold':>10} | {'PCK':>7}")
    print("-" * 22)
    for t, p in zip(results['thresholds'], results['pck_curve']):
        print(f"{t:>10.3f} | {p*100:>6.2f}%")
    print("-" * 22)
    print(f"  AUC: {results['auc']*100:.2f}%")
    print(f"  Gespeichert: {ns.output}")


if __name__ == '__main__':
    main()
