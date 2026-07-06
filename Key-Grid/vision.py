import os
import visualizations as visualizer
import numpy as np
import open3d as o3d

# ── Pfade ──────────────────────────────────────────────────────────────────
DATASET_ROOT = r'C:\Users\xrstu\KeypointNet\pcds\02691156'
KEYPOINTS_FILE = r'results\airplane_v3_k10_keypoints.npy'
OUTPUT_DIR = r'results\visualizations_airplane_v3_k10'
N_KEYPOINTS = 10
MAX_POINTS = 2048

# ── Point Clouds laden ─────────────────────────────────────────────────────
def load_pcd_files(folder, max_points=2048):
    point_clouds = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.pcd')])
    print(f"Loading {len(files)} point clouds...")
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
        except Exception as e:
            print(f"Could not load {fname}: {e}")
    return np.array(point_clouds, dtype=np.float32)

print("Loading point clouds...")
pointcloud = load_pcd_files(DATASET_ROOT, MAX_POINTS)
print(f"Loaded {len(pointcloud)} point clouds")

print("Loading keypoints...")
keypoint = np.load(KEYPOINTS_FILE)
print(f"Keypoints shape: {keypoint.shape}")

# Sicherstellen dass beide gleich lang sind
n = min(len(pointcloud), len(keypoint))
pointcloud = pointcloud[:n]
keypoint = keypoint[:n]
print(f"Visualizing {n} models...")

# ── Visualisierung ─────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(n):
    visualizer.save_kp_and_pc_in_pcd(
        pointcloud[i],
        keypoint[i],
        OUTPUT_DIR,
        save=True,
        name=f"airplane_{i:04d}"
    )
    if (i + 1) % 10 == 0:
        print(f"  Visualized {i+1}/{n}")

print(f"\nFertig! Bilder gespeichert in: {OUTPUT_DIR}\\png")
