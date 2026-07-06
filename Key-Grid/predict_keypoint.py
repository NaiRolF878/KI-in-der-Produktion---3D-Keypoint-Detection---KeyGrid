# -*- coding: utf-8 -*-
"""
Adapted for KeypointNet dataset (.pcd files)
Category: Airplane (02691156)
"""
import torch
import merger.merger_net as merger_net
import numpy as np
import argparse
import open3d as o3d
import os

arg_parser = argparse.ArgumentParser(description="Predictor for Keypoints on KeypointNet Airplane dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='model/airplane.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=8,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument('--dataset-root', type=str,
                        default=r'C:\Users\xrstu\KeypointNet\pcds\02691156',
                        help='Path to the airplane pcd folder.')
arg_parser.add_argument('--output', type=str, default='results/airplane_keypoints.npy',
                        help='Output file for predicted keypoints.')

ns = arg_parser.parse_args()

# Load point clouds
def load_pcd_files(folder, max_points=2048):
    point_clouds = []
    filenames = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
    print(f"Found {len(files)} .pcd files")
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
            filenames.append(fname)
        except Exception as e:
            print(f"Could not load {fname}: {e}")
    return np.array(point_clouds, dtype=np.float32), filenames

print(f"Loading point clouds from: {ns.dataset_root}")
x, filenames = load_pcd_files(ns.dataset_root, ns.max_points)
print(f"Loaded {len(x)} point clouds")

# Load model
print(f"Loading model from: {ns.checkpoint_path}")
net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()

# Predict keypoints
print("Predicting keypoints...")
out_kpcd = []
batch_size = 16
with torch.no_grad():
    for i in range(0, len(x), batch_size):
        batch = torch.FloatTensor(x[i:i+batch_size]).to(ns.device)
        key_points, _ = net(batch, 'False')
        for kp in key_points:
            out_kpcd.append(kp.cpu().numpy())
        print(f"  Processed {min(i+batch_size, len(x))}/{len(x)}")

# Save results
os.makedirs(os.path.dirname(ns.output) if os.path.dirname(ns.output) else '.', exist_ok=True)
keypoints_array = np.array(out_kpcd)
np.save(ns.output, keypoints_array)
print(f"Saved keypoints to: {ns.output}")
print(f"Keypoints shape: {keypoints_array.shape}  (models x keypoints x 3)")
