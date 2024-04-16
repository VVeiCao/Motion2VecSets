import os 
import numpy as np
from renderer import render_depth
from multiprocessing import Pool
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--angle', type=int, default=45)

args = parser.parse_args()
angle = args.angle
setting = f"rotate{angle}"

# change angle from degree to radian
angle = angle * np.pi / 180.0
CPU_COUNT = 24


tasks = []
mesh_home = "/mnt/hdd/DeformingThings4D_manifold/animals"
processed_home = "/mnt/hdd/DeformingThings4D_shape/animals"
output_path = "/mnt/hdd/DeformingThings4D_shape_partial/animals"
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
# setting = "rotate0"
body_ids = os.listdir(mesh_home)

for body_id in tqdm(body_ids):
    body_path = os.path.join(mesh_home, body_id, "mesh_seq")
    meshes = os.listdir(body_path)

    meshes = [mesh for mesh in meshes if mesh.endswith('.obj')]
    meshes.sort()
    for mesh in meshes:
        mesh_path = os.path.join(body_path, mesh)
        # print(mesh_path)
        frame_idx = mesh.split('.')[0]
        action = mesh_path.split('/')[-3]
        processed_npz_path = os.path.join(processed_home, action, "pcl_seq", frame_idx.zfill(8) + ".npz")
        loaded_npz = np.load(processed_npz_path)
        loc = loaded_npz['loc']
        scale = loaded_npz['scale']
        task = {
            'mesh_path': mesh_path,
            'loc': loc,
            'scale': scale,
            'yfov': np.pi/3,
            'angle': angle,
            'setting': setting,
            'seed': 666,
            'output_path': output_path,
            'is_animal': True
        }
        # render_depth(task)
        tasks.append(task)

# raise Exception("Done")
print("Number of tasks: ", len(tasks))

with Pool(CPU_COUNT) as p:
    p.map(render_depth, tasks)