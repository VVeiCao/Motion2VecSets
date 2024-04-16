import os
import numpy as np
import torch
import trimesh
import sys
import open3d as o3d 
sys.path.append('../../..')
sys.path.append('')
from util.console import print
from im2mesh.utils.onet_generator import get_generator

sys.path.append('../network')
import core.network.models_ae as models_ae
import core.network.models_diff as models_diff

T = 17
time_attn = True
is_corr = True

inputs_path = './demo/inputs'
outputs_path = './demo/outputs'

ae_model = "kl_d512_m512_l8"
ae_model_path = './ckpts/DT4D/dt4d_shape_ae.pth'

deform_ae_model = 'kl_d512_m512_l32'
deform_ae_model_path = './ckpts/DT4D/dt4d_deform_ae.pth'

diff_model = 'surf512_edm'
diff_model_path = './ckpts/DT4D/dt4d_shape_diff_sparse.pth'
deform_diff_model_path = './ckpts/DT4D/dt4d_deform_diff_sparse.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Filter non-ply files
inputs = [f"inputs_{i}.ply" for i in range(T)]

all_inputs_points = []

print("Reading input files")

for item in inputs:
    input_file = os.path.join(inputs_path, item)
    
    # o3d read the input file (pcl)
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    all_inputs_points.append(points)
    
# Stack all points into a single array
all_inputs_points = np.stack(all_inputs_points, axis=0)

assert all_inputs_points.shape[0] == len(inputs) == T, "Number of input files does not match number of points"

all_inputs_points = torch.from_numpy(all_inputs_points).float().cuda()
all_inputs_points = all_inputs_points.unsqueeze(0)

generator = get_generator()

shape_ae = models_ae.__dict__[ae_model]()
print("Loading shape ae %s" % ae_model_path)
shape_ae.load_state_dict(torch.load(ae_model_path, map_location='cpu')['model'])
shape_ae.to(device)
shape_ae.eval()

shape_model = models_diff.__dict__[f"{ae_model}_{diff_model}"]()
print("Loading shape dm %s" % diff_model_path)
shape_model.load_state_dict(torch.load(diff_model_path, map_location='cpu')['model'])
shape_model.to(device)
shape_model.eval()

deform_ae = models_ae.__dict__[f"de_{deform_ae_model}"]()
print("Loading deform ae %s" % deform_ae_model_path)
deform_ae.load_state_dict(torch.load(deform_ae_model_path, map_location='cpu')['model'])
deform_ae.to(device)
deform_ae.eval()

deform_model = models_diff.__dict__[f"de_{deform_ae_model}_{diff_model}"](withT=time_attn, is_corr=is_corr)
print("Loading deform dm %s" % deform_diff_model_path)
deform_model.load_state_dict(torch.load(deform_diff_model_path, map_location='cpu')['model'])
deform_model.to(device)
deform_model.eval()

print("Generating deformed meshes")
with torch.no_grad():
    shape_sampled_array = shape_model.sample(device, random=True, cond=all_inputs_points[0]).float() 
    first_frame = generator.generate_from_latent(z = shape_sampled_array[:1], F = shape_ae.decode)
    verts = first_frame.vertices
    faces = first_frame.faces  

    input_src = all_inputs_points[:,:1].repeat(1,T,1,1)
    input_tgt = all_inputs_points[:,:T]
    
    shape_cond = shape_sampled_array[:1,:].repeat(T,1,1).unsqueeze(0)
    
    deform_sampled_array = deform_model.sample(device, 
                                                        cond = None,
                                                        shape_cond = shape_cond,
                                                        cond_src_emb= input_src,
                                                        cond_tgt_emb= input_tgt,
                                                        random = False).squeeze(0)
    
    deformed_verts = deform_ae.decode(deform_sampled_array.float(),torch.tensor(verts).unsqueeze(0).repeat(T,1,1).float().to(device)).squeeze(0)
    
    print("Saving deformed meshes to outputs folder")
    for j in range(deformed_verts.shape[0]):
        defomed_mesh = trimesh.Trimesh(vertices= deformed_verts[j].cpu().numpy(), faces=faces, process=False)
        # Export to outputs
        output_file = os.path.join(outputs_path, f"{inputs[j].split('.')[0]}_deformed.obj")
        defomed_mesh.export(output_file)