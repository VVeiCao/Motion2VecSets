import argparse
import os
import numpy as np
import torch
import trimesh
import datetime
import torch.backends.cudnn as cudnn
import sys
import pandas as pd
import yaml
from munch import Munch
sys.path.append('../../..')
sys.path.append('')
import util.misc as misc
from util.console import print
from im2mesh.eval import MeshEvaluator
from dataset.DT4D import DT4D
from dataset.DFAUST import DFAUST
from im2mesh.utils.onet_generator import get_generator, get_generator_dfaust

sys.path.append('../network')
import core.network.models_ae as models_ae
import core.network.models_diff as models_diff

parser = argparse.ArgumentParser('Latent Diffusion - Final Evaluation', add_help=False)
# Model parameters

parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--test_ui', action='store_true', help='evaluate on UI test set') # By default, US
parser.add_argument('--seq_len', type=int, default=17, help='sequence length for training') # By default, 17
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config_path', default=None, required=True, type=str)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--pin_mem', action='store_true')

args = parser.parse_args()

def eval_oflow_all(
    pcl_tgt, points_tgt, occ_tgt, mesh_t_list, evaluator, corr_project_to_final_mesh, eval_corr=True, return_colored_pcd = False
):
    eval_dict_mean, eval_dict_t = {}, {}
    eval_dict_mesh = {}
    T = pcl_tgt.shape[0]
    # eval IOU and CD
    for t, mesh in enumerate(mesh_t_list):
        _eval_dict_mesh = evaluator.eval_mesh(mesh, pcl_tgt[t], None, points_tgt[t], occ_tgt[t])
        for k, v in _eval_dict_mesh.items():
            # ! Modify here 2021.10.5, skip the normal metrics, to avoid ignoring nan in other metrics
            if k.startswith("normal"):
                continue
            if not np.isnan(v):
                if k not in eval_dict_mesh.keys():
                    eval_dict_mesh[k] = [v]
                else:
                    eval_dict_mesh[k].append(v)
            else:
                raise ValueError("Evaluator meets nan")
    for k, v in eval_dict_mesh.items():
        mean_v = np.array(v).mean()
        eval_dict_mean["{}".format(k)] = mean_v
        for t in range(T):
            eval_dict_t["{}_t{}".format(k, t)] = v[t]
    if eval_corr:
        if not return_colored_pcd:
            # eval correspondence
            eval_dict_corr = evaluator.eval_correspondences_mesh(
                mesh_t_list,
                pcl_tgt,
                project_to_final_mesh=corr_project_to_final_mesh,
                return_colored_pcd = False
            )
        else:
            eval_dict_corr,_ = evaluator.eval_correspondences_mesh(
                mesh_t_list,
                pcl_tgt,
                project_to_final_mesh=corr_project_to_final_mesh,
                return_colored_pcd = True)
            
            colored_pcd = evaluator.vis_chamfer(
                mesh_t_list,
                pcl_tgt)
        
        
        corr_list = []
        for k, v in eval_dict_corr.items():
            t = int(k.split(" ")[1])
            eval_dict_t["corr_l2_t%d" % t] = v
            corr_list.append(v)
        eval_dict_mean["corr_l2"] = np.array(corr_list).mean()
    
    if return_colored_pcd:
        return eval_dict_mean, eval_dict_t, colored_pcd
    else:
        return eval_dict_mean, eval_dict_t

def main():
    # Parse the config file
    print(args)
    cudnn.benchmark = True
    device = torch.device(args.device)
    T = args.seq_len
    config_yaml = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    config_yaml = Munch(config_yaml)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    default_cfg = misc.dummy_cfg()
    config_yaml = misc.merge_munch(config_yaml, default_cfg)
    
    if config_yaml.type == "DT4D":
        dataset_test = DT4D('test_ui' if args.test_ui else 'test_us', cfg=config_yaml)
        generator = get_generator()
    elif config_yaml.type == "DFAUST":
        dataset_test = DFAUST('test_ui' if args.test_ui else 'test_us', cfg=config_yaml)
        generator = get_generator_dfaust()
    else:
        raise ValueError("Invalid dataset type")
    
    shape_ae = models_ae.__dict__[config_yaml.ae_model]()
    print("Loading shape ae %s" % config_yaml.ae_model_path)
    shape_ae.load_state_dict(torch.load(config_yaml.ae_model_path, map_location='cpu')['model'])
    shape_ae.to(device)
    shape_ae.eval()

    shape_model = models_diff.__dict__[f"{config_yaml.ae_model}_{config_yaml.diff_model}"]()
    print("Loading shape dm %s" % config_yaml.diff_model_path)
    shape_model.load_state_dict(torch.load(config_yaml.diff_model_path, map_location='cpu')['model'])
    shape_model.to(device)
    shape_model.eval()

    deform_ae = models_ae.__dict__[f"de_{config_yaml.deform_ae_model}"]()
    print("Loading deform ae %s" % config_yaml.deform_ae_model_path)
    deform_ae.load_state_dict(torch.load(config_yaml.deform_ae_model_path, map_location='cpu')['model'])
    deform_ae.to(device)
    deform_ae.eval()

    deform_model = models_diff.__dict__[f"de_{config_yaml.deform_ae_model}_{config_yaml.diff_model}"](withT=config_yaml.time_attn, is_corr=config_yaml.is_corr)
    print("Loading deform dm %s" % config_yaml.deform_diff_model_path)
    deform_model.load_state_dict(torch.load(config_yaml.deform_diff_model_path, map_location='cpu')['model'])
    deform_model.to(device)
    deform_model.eval()

    evaluator = MeshEvaluator(n_points=100000)

    with torch.no_grad():
        metric_loggers = []
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_loggers.append(metric_logger)
        header = 'Test:'

        data_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            num_workers=16,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )

        formatted_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        exp_name = "partial" if config_yaml.use_depth else "sparse"
        exp_name += "_ui" if args.test_ui else "_us"
        out_path = f"output/metrics/{config_yaml.type}/{exp_name}/exp_{formatted_date}"

        os.makedirs(out_path, exist_ok=True)

        out_file_temp = f"{out_path}/eval.csv"
        out_file_temp_t = f"{out_path}/eval_t.csv"

        with open(f'{out_path}/args.txt', 'w') as f:
            for item in vars(args):
                f.write("%s\n" % getattr(args, item))

        print(f"Start evaluation")

        test_results_m, test_results_t = {}, {}
        
        test_results_m['model'] = []
        test_results_m['start_index'] = []
        
        test_results_t['model'] = []
        test_results_t['start_index'] = []
        
        batch_cnt = 0
        for (data,meta_info) in metric_logger.log_every(data_loader, 1, header):
            batch_cnt += 1
            if batch_cnt % 1 == 0:
                surface = data['surface']
                points = data['points']
                labels = data['labels']
                cond = data['inputs']
                
                surface = surface.to(device, non_blocking=True)
                points = points.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                

                shape_sampled_array = shape_model.sample(device, random=True, cond=cond[0]).float()

                model = meta_info['model'][0]
                start_idx = meta_info['start_idx'][0]

                first_frame = generator.generate_from_latent(z = shape_sampled_array[:1], F = shape_ae.decode)

                verts = first_frame.vertices
                faces = first_frame.faces  

                input_src = cond[:,:1].repeat(1,T,1,1)
                input_tgt = cond[:,:T]
                
                shape_cond = shape_sampled_array[:1,:].repeat(T,1,1).unsqueeze(0)
                
                deform_sampled_array = deform_model.sample(device, 
                                                        cond = None,
                                                        shape_cond = shape_cond,
                                                        cond_src_emb= input_src,
                                                        cond_tgt_emb= input_tgt,
                                                        random = False).squeeze(0)
                
                deformed_verts = deform_ae.decode(deform_sampled_array.float(),torch.tensor(verts).unsqueeze(0).repeat(T,1,1).float().to(device)).squeeze(0)

                deformed_mesh_list = []
                for j in range(deformed_verts.shape[0]):
                    defomed_mesh = trimesh.Trimesh(vertices= deformed_verts[j].cpu().numpy(), faces=faces, process=False)
                    deformed_mesh_list.append(defomed_mesh)


                eval_dict_mean, eval_dict_t, colored_pcd = eval_oflow_all(
                            pcl_tgt=surface.squeeze(0).detach().cpu().numpy(),
                            points_tgt=points.squeeze(0).detach().cpu().numpy(),
                            occ_tgt=labels.squeeze(0).detach().cpu().numpy(),
                            mesh_t_list=deformed_mesh_list,
                            evaluator=evaluator,
                            corr_project_to_final_mesh=False,
                            return_colored_pcd=True
                        )
                
                if config_yaml.vis_input:
                    vis_dir = f'{out_path}/vis/inputs'
                    os.makedirs(vis_dir, exist_ok=True)
                    for i in range(cond[0].shape[0]):
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(cond[0][i].detach().cpu().numpy())
                        o3d.io.write_point_cloud(f'{vis_dir}/{i}_input.ply', pcd)

                
                # error map
                if config_yaml.vis_error_map:
                    vis_dir = f'{out_path}/vis/error_map'
                    os.makedirs(vis_dir, exist_ok=True)
                    for j in range(len(colored_pcd)):
                        defomed_colored_mesh = trimesh.Trimesh(vertices= deformed_verts[j].cpu().numpy(), faces=faces, vertex_colors=colored_pcd[j], process=False)
                        defomed_colored_mesh.export(f'{vis_dir}/{j}_pred.obj')

                # corr
                if config_yaml.vis_corr_map:
                    vis_dir = f'{out_path}/vis/corr_map'
                    os.makedirs(vis_dir, exist_ok=True)
                    for j in range(len(colored_pcd)):
                        vertices = deformed_verts[j].cpu().numpy()
                        num_vertices = len(vertices)
                        colors = np.zeros((num_vertices, 3))
                        colors[:, 0] = np.linspace(0, 1, num_vertices)  # Red channel
                        colors[:, 2] = np.linspace(1, 0, num_vertices)  # Blue channel
                        defomed_colored_mesh = trimesh.Trimesh(vertices= vertices, faces=faces, vertex_colors=colors, process=False)
                        defomed_colored_mesh.export(f'{vis_dir}/{j}_pred.obj')

                for k, v in eval_dict_mean.items():
                    if k not in test_results_m.keys():
                        test_results_m[k] = [v]
                    else:
                        test_results_m[k].append(v)
                test_results_m['model'].append(model)
                test_results_m['start_index'].append(start_idx.item())
                for k, v in eval_dict_t.items():
                    if k not in test_results_t.keys():
                        test_results_t[k] = [v]
                    else:
                        test_results_t[k].append(v)
                test_results_t['model'].append(model)
                test_results_t['start_index'].append(start_idx.item())
                metric_logger.update(iou=eval_dict_mean['iou'].item())
                metric_logger.update(chamfer_L1=eval_dict_mean['chamfer-L1(Onet)'].item())
                metric_logger.update(l2=eval_dict_mean['corr_l2'].item())
                
                #visualize l2 loss colored mesh
                formatted_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                if (batch_cnt % 10) == 0 or batch_cnt == len(data_loader) - 1:
                    # transfer dict to dataframe and save
                    eval_df = pd.DataFrame(test_results_m)
                    eval_df.to_csv(out_file_temp)
                    eval_df_t = pd.DataFrame(test_results_t)
                    eval_df_t.to_csv(out_file_temp_t)
                    eval_df.loc['mean'] = eval_df.mean()
                    eval_df_t.loc['mean'] = eval_df_t.mean()
                    iou_cols = [c for c in  eval_df.columns if 'iou' in c]
                    print('Mean IoU: ', eval_df[iou_cols].mean().mean())
                    chamfer_cols = [c for c in  eval_df.columns if 'chamfer-L1(Onet)' in c]
                    print('Mean Chamfer-L1(Onet): ',  eval_df[chamfer_cols].mean().mean())
                    l2_cols = [c for c in  eval_df.columns if 'l2' in c]
                    print('Mean L2: ',  eval_df[l2_cols].mean().mean())
                    
        print('#' * 20, 'Finish Evaluation', '#' * 20)
        iou_cols = [c for c in  eval_df.columns if 'iou' in c]
        print('Mean IoU: ', eval_df[iou_cols].mean().mean())
        chamfer_cols = [c for c in  eval_df.columns if 'chamfer-L1(Onet)' in c]
        print('Mean Chamfer-L1(Onet): ',  eval_df[chamfer_cols].mean().mean())
        l2_cols = [c for c in  eval_df.columns if 'l2' in c]
        print('Mean L2: ',  eval_df[l2_cols].mean().mean())
        print(f"Results saved to {out_path}")
        
if __name__ == '__main__':
    main()
