
import argparse
import trimesh
import numpy as np
import os
import shutil
import glob
import json
import sys
from multiprocessing import Pool
from multiprocessing import Pool, cpu_count
from functools import partial
sys.path.append('../')
sys.path.append('')

from im2mesh.utils.libmesh import check_mesh_contains

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--ext', type=str, default='obj',
                    help='Extensions for meshes.')


parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')
parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                    help='Whether to save truth values as bit array.')


parser.add_argument('--output_folder', type=str,
                    help='Output path for data.')
parser.add_argument('--pointcloud_size', type=int, default=200000,
                    help='Size of point cloud.')
parser.add_argument('--points_nearsurface_size', type=int, default=200000,
                    help='Size of nearsurface points.')
parser.add_argument('--points_size', type=int, default=200000,
                    help='Size of uniform points.')


def main(args):
    seq_folders = os.listdir(os.path.join(args.in_folder))
    seq_folders = [os.path.join(args.in_folder, folder)
                   for folder in sorted(seq_folders)]
    seq_folders = [folder for folder in seq_folders if os.path.isdir(folder)]
    seq_folders.sort()

    assert args.pointcloud_size == args.points_nearsurface_size

    # check if args.output_folder exist, if not, create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)


    face_idx_dict = get_face_idx(seq_folders, args)
    
    n_workers = cpu_count() 
    with Pool(n_workers) as pool:
        func = partial(process_path, args=args, face_idx_dict=face_idx_dict)
        pool.map(func, seq_folders)


def process_path(in_path, args, face_idx_dict):
    modelname = os.path.basename(in_path)
    model_files = glob.glob(os.path.join(in_path, '*.%s' % args.ext))

    export_nearsurface_points(modelname, sorted(model_files), args, face_idx_dict)
    export_points(modelname, sorted(model_files), args)
    export_pointcloud(modelname, sorted(model_files), args, face_idx_dict)
    


def get_face_idx(seq_folders, args):
    face_idx_dict = {}
    n_workers = cpu_count()
    
    with Pool(n_workers) as pool:
        results = pool.map(partial(process_folder, args=args), seq_folders)
    
    for human_id, n_face, face_idx in results:
        if human_id in face_idx_dict:
            if n_face < face_idx_dict[human_id]["n_face"]:
                face_idx_dict[human_id] = {"n_face": n_face, "face_idx": face_idx}
        else:
            face_idx_dict[human_id] = {"n_face": n_face, "face_idx": face_idx}
    
    with open(os.path.join("DFAUST_mesh", "face_idx_dfaust.json"), 'w') as f:
        json.dump({k: {'n_face': v['n_face'], 'face_idx': v['face_idx'].tolist()} for k, v in face_idx_dict.items()}, f)
    
    return face_idx_dict


def process_folder(p, args):
    modelname = os.path.basename(p)
    human_id, action = modelname.split("_", 1)
    model_files = glob.glob(os.path.join(p, '*.%s' % args.ext))
    min_face_info = {"n_face": float('inf')}
    
    print(f"Processing folder: {p}")
    
    for obj in sorted(model_files):
        mesh = trimesh.load(obj, process=False)
        n_face = mesh.faces.shape[0]
        
        if n_face < min_face_info["n_face"]:
            _, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
            min_face_info = {"n_face": n_face, "face_idx": face_idx}
    
    print(f"Finished processing folder: {p}")
    return human_id, min_face_info["n_face"], min_face_info["face_idx"]


def get_loc_scale(mesh, args):
    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        bbox = mesh.bounding_box.bounds
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

    return loc, scale


# Export functions
def export_pointcloud(modelname, model_files, args, face_idx_dict):
    out_folder = os.path.join(args.output_folder, modelname, "pcl_seq")

    if os.path.exists(out_folder):
        if not args.overwrite:
            print('Pointcloud already exist: %s' % out_folder)
            return
        else:
            shutil.rmtree(out_folder)
            print('Delete existed %s' % out_folder)

    # Create out_folder
    os.makedirs(out_folder)
    number, action = modelname.split("_", 1)
    mesh = trimesh.load(model_files[0], process=False)

    face_idx = face_idx_dict[number]['face_idx']

    alpha = np.random.dirichlet((1,)*3, args.pointcloud_size)

    for it, model_file in enumerate(model_files):
        out_file = os.path.join(out_folder, '%08d.npz' % it)
        mesh = trimesh.load(model_file, process=False)
        loc, scale = get_loc_scale(mesh, args)
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)

        vertices = mesh.vertices
        faces = mesh.faces
        v = vertices[faces[face_idx]]
        points = (alpha[:, :, None] * v).sum(axis=1)

        print('Writing pointcloud: %s' % out_file)
        # Compress
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        points = points.astype(dtype)
        loc = loc.astype(dtype)
        scale = scale.astype(dtype)

        np.savez(out_file, points=points, loc=loc, scale=scale)
        # use the vertices and faces to save the mesh



def export_nearsurface_points(modelname, model_files, args, face_idx_dict, sigma1=0.1, sigma2=0.02):
    out_folder = os.path.join(args.output_folder, modelname, "points_near_seq")

    if os.path.exists(out_folder):
        if not args.overwrite:
            print('Points_nearsurface already exist: %s' % out_folder)
            return
        else:
            shutil.rmtree(out_folder)
            print('Delete existed %s' % out_folder)

    # Create out_folder
    os.makedirs(out_folder)
    number, action = modelname.split("_", 1)
    face_idx = face_idx_dict[number]['face_idx']
    n_points_nearsurface = args.points_nearsurface_size

    alpha = np.random.dirichlet((1,)*3, n_points_nearsurface)
    noise1 = np.random.rand(int(n_points_nearsurface//2), 1) - 0.5
    noise1 = noise1 * sigma1
    noise2 = np.random.rand(int(n_points_nearsurface//2), 1) - 0.5 
    noise2 = noise2 * sigma2
    noise = np.concatenate([noise1, noise2], axis=0)


    for it, model_file in enumerate(model_files):
        out_file = os.path.join(out_folder, '%08d.npz' % it)

        
        mesh = trimesh.load(model_file, process=False)
        if not mesh.is_watertight:
            print('Warning: mesh %s is not watertight!')

        loc, scale = get_loc_scale(mesh, args)
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)


        # _, face_idx = mesh.sample(n_points_nearsurface, return_index=True)
        vertices = mesh.vertices
        faces = mesh.faces
        v = vertices[faces[face_idx]]
        normals = mesh.face_normals[face_idx]
        points = (alpha[:, :, None] * v).sum(axis=1)
        points_nearsuf = points + normals * noise

        occupancies = check_mesh_contains(mesh, points_nearsuf)

        print('Writing points: %s' % out_file)

        # Compress
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        points_nearsuf = points_nearsuf.astype(dtype)
        loc = loc.astype(dtype)
        scale = scale.astype(dtype)

        if args.packbits:
            occupancies = np.packbits(occupancies)

        np.savez(out_file, points=points_nearsuf, occupancies=occupancies,
                    loc=loc, scale=scale)



def export_points(modelname, model_files, args):
    out_folder = os.path.join(args.output_folder, modelname, "points_seq")

    if os.path.exists(out_folder):
        if not args.overwrite:
            print('Points already exist: %s' % out_folder)
            return
        else:
            shutil.rmtree(out_folder)
            print('Delete existed %s' % out_folder)

    os.makedirs(out_folder)

    n_points_uniform = args.points_size

    for it, model_file in enumerate(model_files):
        out_file = os.path.join(out_folder, '%08d.npz' % it)
        mesh = trimesh.load(model_file, process=False)
        if not mesh.is_watertight:
            print('Warning: mesh %s is not watertight!')

        loc, scale = get_loc_scale(mesh, args)
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)

        boxsize = 1 + args.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = boxsize * (points_uniform - 0.5)
        points = points_uniform

        occupancies = check_mesh_contains(mesh, points)
        print('Writing points: %s' % out_file)

        # Compress
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        points = points.astype(dtype)
        loc = loc.astype(dtype)
        scale = scale.astype(dtype)

        if args.packbits:
            occupancies = np.packbits(occupancies)

        np.savez(out_file, points=points, occupancies=occupancies,
                 loc=loc, scale=scale)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)