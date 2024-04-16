# from https://github.com/autonomousvision/occupancy_flow
from torchvision import transforms
import dataset.oflow_dataset as oflow_dataset
from torch.utils.data import Dataset
import logging
import numpy as np
import torch
from util.console import print

class DFAUST(Dataset):
    def __init__(self, split, cfg=None) -> None:
        super().__init__()
        self.split = split.lower()
        
        if self.split == 'test_ui':
            self.split = 'test_new_individual'
        elif self.split == 'test_us':
            self.split = 'test'
        self.cfg = cfg

        if self.cfg.stage == "shape":
            seq_len_train = 1
            seq_len_val = seq_len_train
            self.seq_len = seq_len_train if split == "train" else seq_len_val
            self.n_training_frames = -1
        elif self.cfg.stage == "deform":
            if self.cfg.mode == "ae":
                seq_len_train = 2
                seq_len_val = seq_len_train
                self.seq_len = seq_len_train if split == "train" else seq_len_val
                self.n_training_frames = -1
                self.n_selected_frames = self.cfg.n_selected_frames
                self.n_sample_pro_model = self.cfg.n_sample_pro_model
                self.interval_between_frames = self.cfg.interval_between_frames
            elif self.cfg.mode == "diff":
                self.seq_len = self.cfg.length_sequence
                self.offset_sequence = self.cfg.offset_sequence
                self.n_training_frames = self.cfg.n_training_frames
        else:
            raise NotImplementedError("Not implemented")

        self.pc_size = cfg.pc_size
        self.num_samples = self.pc_size // 2
        
        self.dataset = self.get_dataset()
        logging.info(
            "Use dataset implemented by O-Flow: https://github.com/autonomousvision/occupancy_flow"
        )
        if self.cfg.stage == "shape":
            if split == "train":
                self.transform = AxisScaling((0.75, 1.25), True) if self.cfg.mode == "ae" else None
                self.sampling = True
                self.surface_sampling = True
            elif split == "val":
                self.transform = None
                self.sampling = False
                self.surface_sampling = True
            else:
                self.transform = None
                self.sampling = False
                self.surface_sampling = True
        elif self.cfg.stage == "deform":
            if split == "train":
                self.transform = None
                self.sampling = True
                self.surface_sampling = True
            elif split == "val":
                self.transform = None
                self.sampling = False
                self.surface_sampling = False
            else:
                self.transform = None
                self.sampling = False
                self.surface_sampling = False

    def __len__(self) -> int:
        return len(self.dataset)

    def get_shape_ae(self, index):
        data = self.dataset.__getitem__(index)
        meta_info = self.dataset.models[index]
        if "points" in data.keys():
            if data["points"].ndim == 3:
                try:
                    if self.n_training_frames > 0 and self.split == "train":
                        assert data["points"].shape[0] == self.n_training_frames
                    else:
                        assert data["points"].shape[0] == self.seq_len
                except:
                    raise RuntimeError("Data Length Invalid")
                
        surface = data["pointcloud"]
        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False) # downsample to 2048
            surface = surface[ind]
        surface = torch.from_numpy(surface)

        vol_points = data["points"] # uniform points
        vol_label = data["points.occ"] # 0: outside, 1: inside
        near_points = data['points_nearsuf'] # near surface points
        near_label = data['points_nearsuf.occ'] # 0: outside, 1: inside

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False) # downsample to 1024
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)# downsample to 1024
            near_points = near_points[ind]
            near_label = near_label[ind]

        
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0) # uniform sampled points + surface and near surface points 2048
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)
        
        model_data = {}
        model_data["points"] = points
        model_data["labels"] = labels
        model_data["surface"] = surface
        
        return model_data, meta_info
    
    def get_shape_diff(self, index):
        data = self.dataset.__getitem__(index)
        meta_info = self.dataset.models[index]
        viz_id = "{}_".format(index)
        for v in meta_info.values():
            viz_id += str(v) + "_"
        meta_info["viz_id"] = viz_id
        if "points" in data.keys():
            if data["points"].ndim == 3:
                try:
                    if self.n_training_frames > 0 and self.split == "train":
                        assert data["points"].shape[0] == self.n_training_frames
                    else:
                        assert data["points"].shape[0] == self.seq_len
                except:
                    print(data["points"].shape[0])
                    raise RuntimeError("Data Length Invalid")
                
        inputs = data["inputs"]
        inputs = inputs.squeeze(0)
        inputs = torch.from_numpy(inputs)

        surface = data["pointcloud"]
        surface = surface.squeeze(0)

        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False) # downsample to 2048
            surface = surface[ind,:]

        vol_points = data["points"] # uniform points
        vol_label = data["points.occ"] # 0: outside, 1: inside
        near_points = data['points_nearsuf'] # near surface points
        near_label = data['points_nearsuf.occ'] # 0: outside, 1: inside

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False) # downsample to 1024
            vol_points = vol_points[ind,:]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)# downsample to 1024
            near_points = near_points[ind,:]
            near_label = near_label[ind]

        
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0) # uniform sampled points + surface and near surface points 2048
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)
        
        model_data = {}

        model_data['surface'] = surface
        model_data['points'] = points
        model_data['labels'] = labels
        model_data['inputs'] = inputs

        return model_data, meta_info
    
    def get_deform_ae(self, index):
        data = self.dataset.__getitem__(index)
        meta_info = self.dataset.models[index]
        viz_id = "{}_".format(index)
        for v in meta_info.values():
            viz_id += str(v) + "_"
        meta_info["viz_id"] = viz_id
        if "points" in data.keys():
            if data["points"].ndim == 3:
                try:
                    if self.n_training_frames > 0 and self.split == "train":
                        assert data["points"].shape[0] == self.n_training_frames
                    else:
                        assert data["points"].shape[0] == self.seq_len
                except:
                    print(data["points"].shape[0])
                    raise RuntimeError("Data Length Invalid")
                
        assert len(data['points_nearsuf']) == 2
                
        surface = data["pointcloud"]

        ind = np.random.default_rng().choice(surface.shape[1], self.pc_size, replace=False)
        ind2 = np.random.default_rng().choice(surface.shape[1], self.pc_size, replace=False)
        
        surface_src_corr = surface[0,ind,:]
        surface_tgt_corr = surface[1,ind,:]
        
        surface_src = surface[0,ind,:]
        surface_tgt = surface[1,ind2,:]
        
        surface_src = torch.from_numpy(surface_src)
        surface_tgt = torch.from_numpy(surface_tgt)
        
        surface_src_corr = torch.from_numpy(surface_src_corr)
        surface_tgt_corr = torch.from_numpy(surface_tgt_corr)

        near_points = data['points_nearsuf'] # near surface points

        ind = np.random.default_rng().choice(near_points.shape[1], self.num_samples, replace=False)# downsample to 1024
        near_points = near_points[:,ind,:]
        near_points_src = near_points[0,:,:]
        near_points_tgt = near_points[1,:,:]
        
        near_points_src = torch.from_numpy(near_points_src)
        near_points_tgt = torch.from_numpy(near_points_tgt)
        
        points_src = torch.cat([surface_src_corr, near_points_src], dim=0)
        points_tgt = torch.cat([surface_tgt_corr, near_points_tgt], dim=0)

        model_data = {}
  
        model_data["points_src"] = points_src
        model_data["points_tgt"] = points_tgt
        model_data["surface_src"] = surface_src
        model_data["surface_tgt"] = surface_tgt

        return model_data, meta_info
    
    def get_deform_diff(self, index):
        data = self.dataset.__getitem__(index)
        meta_info = self.dataset.models[index]
        viz_id = "{}_".format(index)
        for v in meta_info.values():
            viz_id += str(v) + "_"
        meta_info["viz_id"] = viz_id
        model_data = {}
        
        if "test" in self.split: # Eval
            model_data['points'] = data['points']
            model_data['labels'] = data['points.occ']
            model_data["surface"] = data["pointcloud"]
            model_data["inputs"] = data["inputs"]
        else: 
            surface = data["pointcloud"]

            ind = np.random.default_rng().choice(surface.shape[1], self.pc_size, replace=False)
            surface = surface[:,ind,:]

            random_idx = np.random.choice(replace=False, a=surface.shape[0], size=self.n_training_frames)
            random_idx = np.sort(random_idx)

            surface_tgt = surface[random_idx,:,:]

            surface_src = surface[:1,:,:].repeat(self.n_training_frames, axis=0)

            surface_src = torch.from_numpy(surface_src)
            surface_tgt = torch.from_numpy(surface_tgt)

            near_points = data['points_nearsuf'] # near surface points

            ind = np.random.default_rng().choice(near_points.shape[1], self.num_samples, replace=False)# downsample to 1024
            near_points = near_points[:,ind,:]


            near_points_tgt = near_points[random_idx,:,:]
            near_points_src = near_points[:1,:,:].repeat(self.n_training_frames, axis=0)

            near_points_src = torch.from_numpy(near_points_src)
            near_points_tgt = torch.from_numpy(near_points_tgt)

            
            points_src = torch.cat([surface_src, near_points_src], dim=1)
            points_tgt = torch.cat([surface_tgt, near_points_tgt], dim=1)

            if self.transform:
                surface, points = self.transform(surface, points)
            
            model_data["points_src"] = points_src
            model_data["points_tgt"] = points_tgt
            model_data["surface_src"] = surface_src
            model_data["surface_tgt"] = surface_tgt

            inputs = data["inputs"]

            inputs_src = inputs[:1,:,:].repeat(self.n_training_frames, axis=0)
            inputs_tgt = inputs[random_idx,:,:]

            inputs_src = torch.from_numpy(inputs_src)
            inputs_tgt = torch.from_numpy(inputs_tgt)

            model_data["inputs_src"] = inputs_src
            model_data["inputs_tgt"] = inputs_tgt

        return model_data, meta_info
    
    def __getitem__(self, index: int):
        
        if self.cfg.stage == "shape":
            if self.cfg.mode == "ae":
                return self.get_shape_ae(index)
            elif self.cfg.mode == "diff":
                return self.get_shape_diff(index)
            else:
                raise NotImplementedError("Not implemented")
        elif self.cfg.stage == "deform":
            if self.cfg.mode == "ae":
                return self.get_deform_ae(index)
            elif self.cfg.mode == "diff":
                return self.get_deform_diff(index)
            else:
                raise NotImplementedError("Not implemented")

    def get_dataset(self, return_idx=False, return_category=False):
        """Returns the dataset.

        Args:
            model (nn.Module): the model which is used
            cfg (dict): config dictionary
            return_idx (bool): whether to include an ID field
        """
        # dataset_type = cfg.type
        dataset_folder = self.cfg.dataset_path
        
        if self.cfg.stage == "shape":
            categories = ['data_processed_shape']
        elif self.cfg.stage == "deform":
            categories = ['data_processed_deform']
            
        # Get split
        fields = self.get_data_fields()
        # Input fields
        
        if self.cfg.mode == "diff":
            inputs_field = self.get_inputs_field()
        else:
            inputs_field = None
            
        if inputs_field is not None:
            fields["inputs"] = inputs_field

        if return_idx:
            fields["idx"] = oflow_dataset.IndexField()

        if return_category:
            fields["category"] = oflow_dataset.CategoryField()
        if self.cfg.stage == "shape" and self.cfg.mode == "ae":
            dataset = oflow_dataset.HumansDataset_shape(
                dataset_folder,
                fields,
                split=self.split,
                categories=categories,
                length_sequence=self.seq_len,
                n_files_per_sequence=-1,
                offset_sequence=0,
                ex_folder_name='pcl_seq',
            )
        elif self.cfg.stage == "shape" and self.cfg.mode == "diff":
            dataset = oflow_dataset.HumansDataset_shape_diffusion(
                dataset_folder,
                fields,
                split=self.split,
                categories=categories,
                length_sequence=self.seq_len,
                n_files_per_sequence=-1,
                offset_sequence=0,
                ex_folder_name='pcl_seq',
            )
        elif self.cfg.stage == "deform" and self.cfg.mode == "ae":
            dataset = oflow_dataset.HumansDataset_deform_wo_cano(
                dataset_folder,
                fields,
                split=self.split,
                categories=categories,
                length_sequence=self.seq_len,
                n_files_per_sequence=-1,
                offset_sequence=0,
                ex_folder_name='pcl_seq',
                n_sample_pro_model=self.n_sample_pro_model,
                interval_between_frames=self.interval_between_frames,
                n_selected_frames=self.n_selected_frames,
                repeat=self.cfg.repeat
            )
        elif self.cfg.stage == "deform" and self.cfg.mode == "diff":
            dataset = oflow_dataset.HumansDataset_deform_wo_cano_diffusion(
                dataset_folder,
                fields,
                split=self.split,
                categories=categories,
                length_sequence=self.seq_len,
                n_files_per_sequence=-1,
                offset_sequence=self.offset_sequence,
                ex_folder_name='pcl_seq'
            )
        return dataset

    def get_data_fields(self):
        """Returns data fields.

        Args:
            split (str): split (train|val|test_ui|test_us)
            cfg (yaml config): yaml config object
        """
        fields = {}

        p_folder = 'points_seq'
        p_nearsuface_folder = 'points_near_seq'
        pcl_folder = 'pcl_seq'
        dep_folder = 'depth_rotate45'
        mesh_folder = ''
        generate_interpolate = False
        unpackbits = True
        training_all = True
        n_training_frames = -1
        transf_pt, transf_pt_val, transf_pcl, transf_pcl_val = None, None, None, None
        pts_iou_field = oflow_dataset.PointsSubseqField
        pts_corr_field = oflow_dataset.PointCloudSubseqField

        # not_choose_last = False
        training_multi_files = False
        all_steps = True
        
        if self.cfg.stage == "shape" and self.cfg.mode == "diff":
            all_steps = False

        
        if self.split == "train":
            if training_all:
                if self.cfg.stage == "shape":
                    fields["points"] = pts_iou_field(
                        p_folder,
                        transform=transf_pt,
                        all_steps=all_steps,
                        seq_len=self.seq_len,
                        unpackbits=unpackbits,
                        use_multi_files=training_multi_files,
                    )
                fields["points_nearsuf"] = pts_iou_field(
                    p_nearsuface_folder,
                    transform=transf_pt,
                    all_steps=all_steps,
                    seq_len=self.seq_len,
                    unpackbits=unpackbits,
                    use_multi_files=training_multi_files,
                )
            else:
                if self.cfg.stage == "shape":
                    fields["points"] = pts_iou_field(
                        p_folder,
                        sample_nframes=n_training_frames,
                        transform=transf_pt,
                        seq_len=self.seq_len,
                        fixed_time_step=0,
                        unpackbits=unpackbits,
                        use_multi_files=training_multi_files,
                    )
                fields["points_nearsuf"] = pts_iou_field(
                    p_nearsuface_folder,
                    sample_nframes=n_training_frames,
                    transform=transf_pt,
                    seq_len=self.seq_len,
                    fixed_time_step=0,
                    unpackbits=unpackbits,
                    use_multi_files=training_multi_files,
                )
        # only training can be boost by multi-files
        # modify here, if not train, val should also load the same as the test
        else:
            fields["points"] = pts_iou_field(
                p_folder,
                transform=transf_pt_val,
                all_steps=all_steps,
                seq_len=self.seq_len,
                unpackbits=unpackbits,
            )
            fields["points_nearsuf"] = pts_iou_field(
                p_nearsuface_folder,
                transform=transf_pt_val,
                all_steps=all_steps,
                seq_len=self.seq_len,
                unpackbits=unpackbits,
            )
            fields[
                "points_mesh"
            ] = pts_corr_field(  # ? this if for correspondence? Checked, this is for chamfer distance, make sure that because here we use tranforms, teh pts in config file must be 100000
                pcl_folder, transform=transf_pcl_val, seq_len=self.seq_len
            )
            
        fields["pointcloud"] = pts_corr_field(
            pcl_folder,
            transform=transf_pcl,
            seq_len=self.seq_len,
            use_multi_files=training_multi_files,
        )
        
        fields["depth"] = pts_corr_field(
            dep_folder,
            transform=transf_pcl,
            seq_len=self.seq_len,
            use_multi_files=training_multi_files
        )
        
        if "test" in self.split and generate_interpolate:
            fields["mesh"] = oflow_dataset.MeshSubseqField(
                mesh_folder, seq_len=self.seq_len, only_end_points=True
            )
        fields["oflow_idx"] = oflow_dataset.IndexField()
        return fields

    def get_transforms(self):
        """Returns transform objects.

        Args:
            cfg (yaml config): yaml config object
        """
        n_pcl = self.cfg.pc_size
        n_pt = n_pcl // 2
        n_pt_eval = 10000

        transf_pt = oflow_dataset.SubsamplePoints(n_pt)
        transf_pt_val = oflow_dataset.SubsamplePoints(n_pt_eval)
        transf_pcl_val = oflow_dataset.SubsamplePointcloudSeq(n_pt_eval, random=False)
        transf_pcl = oflow_dataset.SubsamplePointcloudSeq(n_pcl, connected_samples=True)

        return transf_pt, transf_pt_val, transf_pcl, transf_pcl_val

    def get_inputs_field(self):
        """Returns the inputs fields.

        Args:
            split (str): the mode which is used
            cfg (dict): config dictionary
        """
        connected_samples = True
        transform = transforms.Compose(
            [
                oflow_dataset.SubsamplePointcloudSeq(
                    self.cfg.n_inputs,
                    connected_samples=connected_samples,
                ),
                oflow_dataset.PointcloudNoise(
                    self.cfg.inputs_noise_std
                ),
            ]
        )
        training_multi_files = False

        target_seq = 'pcl_seq' # By default
        if self.cfg.mode == "diff":
            if self.cfg.use_half and self.cfg.use_depth:
                raise ValueError("This two options are not compatible")
            if self.cfg.use_half:
                target_seq = 'half_seq'
            if self.cfg.use_depth:
                target_seq = 'depth_rotate45'

        inputs_field = oflow_dataset.PointCloudSubseqField(
            target_seq,
            transform=transform,
            seq_len=self.seq_len,
            use_multi_files=training_multi_files,
        )
        
        return inputs_field

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (0.5 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-0.5, max=0.5)

        return surface, point