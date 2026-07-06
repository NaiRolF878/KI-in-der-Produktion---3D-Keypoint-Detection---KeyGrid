# -*- coding: utf-8 -*-
"""
Adapted for KeypointNet dataset (.pcd files)
Based on original predictor_keypointnet.py by eliphat
"""
import torch
import merger.merger_net as merger_net
import json
import tqdm
import numpy as np
import argparse
import os

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\annotations\airplane.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds',
                        help='Point cloud root folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='model/sm_airplane.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-p', '--prediction-output', type=str, default='results/sm_airplane_keypoints.npz',
                        help='Output file where prediction results are written.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')

ns = arg_parser.parse_args()

os.makedirs(os.path.dirname(ns.prediction_output) if os.path.dirname(ns.prediction_output) else '.', exist_ok=True)

net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device), weights_only=False)['model_state_dict'])
net.eval()


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float32)
    return pc


print(f"Loading annotations from: {ns.annotation_json}")
kpn_ds = json.load(open(ns.annotation_json))
print(f"Found {len(kpn_ds)} entries")

out_kpcd = []
out_nfact = []

for i in tqdm.tqdm(range(0, len(kpn_ds), ns.batch), unit_scale=ns.batch):
    Q = []
    for j in range(ns.batch):
        if i + j >= len(kpn_ds):
            continue
        entry = kpn_ds[i + j]
        cid = entry['class_id']
        mid = entry['model_id']
        pcd_file = r'{}/{}/{}.pcd'.format(ns.pcd_path, cid, mid)
        if not os.path.exists(pcd_file):
            continue
        pc = naive_read_pcd(pcd_file)
        if len(pc) == 0:
            continue
        # Normalize
        pcmax = pc.max()
        pcmin = pc.min()
        pcn = (pc - pcmin) / (pcmax - pcmin + 1e-8)
        pcn = 2.0 * (pcn - 0.5)
        # Ensure 2048 points
        if len(pcn) >= ns.max_points:
            idx = np.random.choice(len(pcn), ns.max_points, replace=False)
        else:
            idx = np.random.choice(len(pcn), ns.max_points, replace=True)
        Q.append(pcn[idx])
        out_nfact.append([pcmax, pcmin])

    if len(Q) == 0:
        continue
    if len(Q) == 1:
        Q.append(Q[-1])
        out_nfact.append(out_nfact[-1])

    with torch.no_grad():
        recon, key_points, kpa, emb, null_activation = net(torch.Tensor(np.array(Q)).to(ns.device))
    for kp in key_points:
        out_kpcd.append(kp)

for i in range(len(out_kpcd)):
    out_kpcd[i] = out_kpcd[i].cpu().numpy()

np.savez(ns.prediction_output, kpcd=out_kpcd, nfact=out_nfact)
print(f"Saved keypoints to: {ns.prediction_output}")
print(f"Keypoints shape: {np.array(out_kpcd).shape}  (models x keypoints x 3)")
