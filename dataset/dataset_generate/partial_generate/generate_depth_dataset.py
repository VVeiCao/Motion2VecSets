import os 
import numpy as np
from renderer import render_depth
from multiprocessing import Pool
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--angle', type=int, default=0)

args = parser.parse_args()
angle = args.angle
setting = f"rotate{angle}"

# change angle from degree to radian
angle = angle * np.pi / 180.0
CPU_COUNT = 16


tasks = []
mesh_home = "/mnt/hdd/fintune_base/dfaust_mesh/data"
processed_home = "/mnt/hdd/fintune_base/Humans/D-FAUST"
output_path = "/mnt/hdd/fintune_base/dfaust_partial"
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
# setting = "rotate0"
body_ids = os.listdir(mesh_home)

for body_id in tqdm(body_ids):
    body_path = os.path.join(mesh_home, body_id)
    actions = os.listdir(body_path)
    for action in actions:
        action_path = os.path.join(body_path, action)
        meshes = os.listdir(action_path)
        meshes = [mesh for mesh in meshes if mesh.endswith('.obj')]
        meshes.sort()
        for mesh in meshes:
            mesh_path = os.path.join(action_path, mesh)
            # print(mesh_path)
            frame_idx = mesh.split('.')[0]
            body_id = mesh_path.split('/')[-3]
            action = mesh_path.split('/')[-2]
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
                'output_path': output_path
            }
            # render_depth(task)
            tasks.append(task)

print("Number of tasks: ", len(tasks))

with Pool(CPU_COUNT) as p:
    p.map(render_depth, tasks)