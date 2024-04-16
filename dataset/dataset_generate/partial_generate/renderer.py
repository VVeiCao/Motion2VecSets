import sys
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' # 'osmesa', 'egl'
import numpy as np
import pyrender
import trimesh
import cv2
import open3d as o3d

def render_depth(task_dict, write_depth=False, write_color=True, write_pcd=False):
    print("Rendering depth for: ", task_dict['mesh_path'])
    # assert 'mesh_path', 'loc', 'scale' should be in task_dict
    assert 'mesh_path' in task_dict
    assert 'loc' in task_dict
    assert 'scale' in task_dict
    assert 'yfov' in task_dict
    assert 'angle' in task_dict
    assert 'setting' in task_dict
    assert 'output_path' in task_dict
    assert 'seed' in task_dict
    assert 'is_animal' in task_dict
    is_animal = task_dict['is_animal']
    if is_animal:
        W = 2000
        H = 2000
    else:
        W = 1600
        H = 1600

    # load mesh
    org_mesh = trimesh.load(task_dict['mesh_path'])

    frame_idx = task_dict['mesh_path'].split('/')[-1].split('.')[0]
    action = task_dict['mesh_path'].split('/')[-3]

    loc = task_dict['loc']
    scale = task_dict['scale']

    org_mesh.apply_translation(-loc)
    org_mesh.apply_scale(1.0/scale)

    scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02],
                        bg_color=[1.0, 1.0, 1.0])
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    
    yfov = task_dict['yfov']
    cam = pyrender.PerspectiveCamera(yfov=yfov)
    cam_pose = np.eye(4)
    if is_animal:
        cam_pose[:3, 3] = np.array([0, 0, 1.0])
    else:
        cam_pose[:3, 3] = np.array([0, 0, 1.25])

    mesh = pyrender.Mesh.from_trimesh(org_mesh)

    angle = task_dict['angle']
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((0, angle, 0))
    if is_animal:
        rotation_x_90 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
        rotation = np.dot(rotation, rotation_x_90)

    node_light = pyrender.Node(light=light, matrix=cam_pose)
    node_cam = pyrender.Node(camera=cam, matrix=cam_pose)

    m_pose = np.eye(4)
    m_pose[:3, 3] = np.array([0, 0, 0])
    m_pose[:3, :3] = rotation
    node_mesh = pyrender.Node(mesh=mesh, matrix=m_pose)
    scene.add_node(node_cam)
    scene.add_node(node_light)
    scene.add_node(node_mesh)
    
    renderer = pyrender.OffscreenRenderer(W, H)
    color, depth = renderer.render(scene)
    if write_depth:
        # Save depth
        output_folder_depth = os.path.join(f"{task_dict['output_path']}_depth", action, f'depth_{task_dict["setting"]}')
        if not os.path.exists(output_folder_depth):
            os.makedirs(output_folder_depth, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder_depth, f'{frame_idx.zfill(8)}.png'), depth)
    if write_color:
        # Save color
        output_folder_color = os.path.join(f"{task_dict['output_path']}_color", action, f'depth_{task_dict["setting"]}')
        if not os.path.exists(output_folder_color):
            os.makedirs(output_folder_color, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder_color, f'{frame_idx.zfill(8)}.png'), color)

    fx = (W / 2) / np.tan(yfov / 2)
    fy = (H / 2) / np.tan(yfov / 2)
    cx = W / 2
    cy = H / 2

    depth = depth.reshape(W, H)
    xs, ys = np.where(depth != 0)
        
    pts_3d = []
    for x, y in zip(xs, ys):
        x_3d = (x - cx) * depth[x, y] / fx
        y_3d = (y - cy) * depth[x, y] / fy
        z_3d = depth[x, y] 
        pts_3d.append([y_3d, -x_3d, z_3d])

        
    pts_3d = np.array(pts_3d)
    # save obj
    pts_3d = pts_3d - cam_pose[:3, 3]
    # Flip z axis -
    pts_3d[:, 2] = -pts_3d[:, 2]
    pts_3d = np.dot(pts_3d, rotation)
    print("Number of points: ", pts_3d.shape[0])
    # Sample 50000 points
    if pts_3d.shape[0] > 100000:
        np.random.seed(int(task_dict['seed']))
        pts_3d = pts_3d[np.random.choice(pts_3d.shape[0], 100000, replace=False), :]
    else:
        # Save this mesh_path to a txt file (failed log)
        with open(os.path.join(task_dict['output_path'], 'failed_log.txt'), 'a') as f:
            f.write(task_dict['mesh_path'] + " failed with number of points: " + str(pts_3d.shape[0]) + "\n")


    if write_pcd:
        o3d_pts = o3d.geometry.PointCloud()
        o3d_pts.points = o3d.utility.Vector3dVector(pts_3d)

        output_folder_pcd = os.path.join(f"{task_dict['output_path']}_pcd", action, f'depth_{task_dict["setting"]}')

        if not os.path.exists(output_folder_pcd):
            os.makedirs(output_folder_pcd, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(output_folder_pcd, f'{frame_idx.zfill(8)}.ply'), o3d_pts)
        print("Outputting pcd to: ", os.path.join(output_folder_pcd, f'{frame_idx.zfill(8)}.ply'))

    output_folder = os.path.join(task_dict['output_path'], action, f'depth_{task_dict["setting"]}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    pts_3d = pts_3d.astype(np.float16)
    # np.save(os.path.join(output_folder, f'{frame_idx.zfill(8)}.npy'), pts_3d)
    print("Outputting npy to: ", os.path.join(output_folder, f'{frame_idx.zfill(8)}.npy'))
    np.savez(os.path.join(output_folder, f'{frame_idx.zfill(8)}.npz'), points=pts_3d, loc=loc, scale=scale)