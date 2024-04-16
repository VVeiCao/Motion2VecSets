import torch
import numpy as np
import os
from util.console import print

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
    
class DT4D(torch.utils.data.Dataset):
    def __init__(self, split, cfg):
        self.dataset_path = os.path.join(cfg.dataset_path, 'data_processed_shape' if cfg.stage =='shape' else 'data_processed_deform')
        self.split = split
        self.stage = cfg.stage # 'shape' or 'deform'
        self.mode = cfg.mode # 'ae' or 'diff'
        self.repeat = cfg.repeat
        self.seed = cfg.seed 
        self.use_depth = cfg.use_depth
        self.pc_size = cfg.pc_size
        self.num_samples = cfg.num_samples
        self.n_inputs = cfg.n_inputs
        self.inputs_noise_std = cfg.inputs_noise_std
        self.n_sample_pro_model = cfg.n_sample_pro_model
        self.interval_between_frames = cfg.interval_between_frames
        self.n_selected_frames = cfg.n_selected_frames
        self.length_sequence = cfg.length_sequence
        self.n_files_per_sequence = cfg.n_files_per_sequence
        self.offset_sequence = cfg.offset_sequence
        self.n_training_frames = cfg.n_training_frames
        
        if self.split.find('train') == -1:
            self.repeat = 1
            
        self.split_file_path = os.path.join(self.dataset_path, self.split + '.lst')
        # Do not add if line.len() == 0
        self.split_lst = [line.rstrip('\n') for line in open(self.split_file_path)]
        
        # Sort by name
        self.split_lst.sort()
        self.split_lst_save = self.split_lst.copy()
        self.prepare_data_lst()
        
        if self.split == 'train': # train
            self.transform = AxisScaling((0.75, 1.25), True)
            if self.mode == 'diff':
                self.transform = None
            self.sampling = True
            self.surface_sampling = True
        else: # val and test
            self.transform = None
            self.sampling = False
            self.surface_sampling = True
        
    def prepare_data_lst(self):
        self.split_lst_path = [os.path.join(self.dataset_path, line) for line in self.split_lst]

        self.split_lst_pcl_path = [os.path.join(self.dataset_path, line, 'pcl_seq') for line in self.split_lst]
        self.split_lst_pcl_path = [sorted(os.listdir(line)) for line in self.split_lst_pcl_path]
        self.split_lst_pcl_path = [[os.path.join(self.dataset_path, self.split_lst[i], 'pcl_seq', line) for line in self.split_lst_pcl_path[i]] for i in range(len(self.split_lst_pcl_path))]
        # Merge list
        self.split_lst_pcl_path = [item for sublist in self.split_lst_pcl_path for item in sublist]
        self.split_lst_points_path = [line.replace('pcl_seq', 'points_seq') for line in self.split_lst_pcl_path]
        self.split_lst_points_near_path = [line.replace('pcl_seq', 'points_near_seq') for line in self.split_lst_pcl_path]
        self.split_lst_depth_pat = []
        if self.use_depth:
            self.split_lst_depth_path = [line.replace('pcl_seq', f'depth_rotate45') for line in self.split_lst_pcl_path]

        if self.repeat > 1:
            self.split_lst_pcl_path = self.split_lst_pcl_path * self.repeat
            self.split_lst_points_path = self.split_lst_points_path * self.repeat
            self.split_lst_points_near_path = self.split_lst_points_near_path * self.repeat
            self.split_lst_depth_path = self.split_lst_depth_path * self.repeat
            
        self.split_lst_pcl_len = [len(os.listdir(os.path.join(self.dataset_path, line, 'pcl_seq'))) for line in self.split_lst]
        self.split_lst_near_len = [len(os.listdir(os.path.join(self.dataset_path, line, 'points_near_seq'))) for line in self.split_lst]

        self.models = []
        if self.stage == 'deform':
            if self.mode == 'ae':
                models_out, indices_out = self.random_sample_indexes(self.split_lst_save * self.repeat, self.split_lst_pcl_len * self.repeat)
                self.models += [
                    {"model": m, "indices": indices_out[i]} for i, m in enumerate(models_out)
                ]
            elif self.mode == 'diff':
                models_out, indices_out = self.subdivide_into_sequences(self.split_lst_save * self.repeat, self.split_lst_pcl_len * self.repeat)
                self.models += [
                    {"model": m, "start_idx": indices_out[i]} for i, m in enumerate(models_out)
                ]
            else:
                raise NotImplementedError("mode not implemented")
            
        print(f"DT4D {self.stage} dataset is loaded :smile:")
        print("Current stage: [bold red]{}[/bold red]".format(self.stage))
        print("Current mode: [bold red]{}[/bold red]".format(self.mode))
        print("Current split: [bold red]{}[/bold red]".format(self.split))
        print("Dataset is repeated [bold cyan]{}[/bold cyan] times".format(self.repeat))
        print("Depth is used: {}".format(self.use_depth), "with camera angle: {}".format(45))
        print("Length of split: {}".format(len(self.split_lst) if self.stage == 'shape' else len(self.models)))
    
    def __getitem__(self, index):
        if self.stage == 'shape':
            if self.mode == 'ae':
                return self.get_shape_ae(index)
            elif self.mode == 'diff':
                return self.get_shape_diff(index)
            else:
                raise NotImplementedError("mode not implemented")
        elif self.stage == 'deform':
            if self.mode == 'ae':
                return self.get_deform_ae(index)
            elif self.mode == 'diff':
                return self.get_deform_diff(index)
            else:
                raise NotImplementedError("mode not implemented")
    
    def __len__(self):
        if self.stage == 'shape':
            return sum(self.split_lst_pcl_len) * self.repeat 
        elif self.stage == 'deform':
            return len(self.models)
        else:
            raise NotImplementedError("stage not implemented")
    
    def random_sample_indexes(self, models, models_len):
        n_sample_pro_model = self.n_sample_pro_model
        interval_between_frames = self.interval_between_frames
        n_selected_frames = self.n_selected_frames

        # Initialize output lists
        models_out = []
        indexes_out = []

        # Loop over each model
        for idx, model in enumerate(models):
            # For each sample per model
            for n in range(n_sample_pro_model):
                # Initialize indices list for current sample
                indexes = []

                # Select n_selected_frames number of indices
                for i in range(n_selected_frames):
                    # If first index, randomly select from range
                    if i == 0:
                        indexes.append(np.random.randint(0, models_len[idx] - interval_between_frames))
                    else:
                        # For subsequent indices, select within interval_between_frames from the previous index
                        indexes.append(indexes[-1] + np.random.randint(0, interval_between_frames))
                    
                # Append the selected indices and corresponding model to output lists
                indexes_out.append(sorted(indexes))
                models_out.append(model)
        
        return models_out, indexes_out  
    
    def subdivide_into_sequences(self, models, models_len):
        """Subdivides model sequence into smaller sequences.

        Args:
            models (list): list of model names
            models_len (list): list of lengths of model sequences
        """
        length_sequence = self.length_sequence
        n_files_per_sequence = self.n_files_per_sequence
        offset_sequence = self.offset_sequence

        # Remove files before offset
        models_len = [l - offset_sequence for l in models_len]

        # Reduce to maximum number of files that should be considered
        if n_files_per_sequence > 0:
            models_len = [min(n_files_per_sequence, l) for l in models_len]

        models_out = []
        start_idx = []
        for idx, model in enumerate(models):
            for n in range(0, models_len[idx] - length_sequence + 1):
                models_out.append(model)
                start_idx.append(n + offset_sequence)

        return models_out, start_idx

    def get_shape_ae(self, index):
        pcl_path = self.split_lst_pcl_path[index]
        points_path = self.split_lst_points_path[index]
        points_near_path = self.split_lst_points_near_path[index]
        
        meta_info = {
            "pcl_file": pcl_path,
            "points_file": points_path,
            "points_nearsurface_file": points_near_path,
        }        
        
        pcl = np.load(pcl_path)
        points = np.load(points_path)
        points_nearsurface = np.load(points_near_path)
        
        surface = pcl['points']
        
        src = surface
        
        # Val and Test will do surface sampling also
        if self.surface_sampling:
            if self.split.find('train') == -1: # val and test
                ind = np.random.default_rng(seed=self.seed).choice(src.shape[0], self.pc_size, replace=False)
            ind = np.random.default_rng().choice(src.shape[0], self.pc_size, replace=False)
            src = src[ind]
            
        src = torch.from_numpy(src).float()
        vol_points = points['points']
        vol_label = np.unpackbits(points['occupancies'])

        near_points = points_nearsurface['points']
        near_label = np.unpackbits(points_nearsurface['occupancies'])
        
        if self.sampling: # val and test won't do sampling
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False) # downsample to 1024
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)# downsample to 1024
            near_points = near_points[ind]
            near_label = near_label[ind]
            
        vol_points = torch.from_numpy(vol_points).float()
        vol_label = torch.from_numpy(vol_label).float()

        if self.split.find('train') != -1: # train
            near_points = torch.from_numpy(near_points).float()
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0) # uniform sampled points + surface and near surface points 2048
            labels = torch.cat([vol_label, near_label], dim=0)
        else: # val and test
            points = vol_points
            labels = vol_label
        if self.transform:
            src, points = self.transform(src, points)
        
        model_data = {}
        model_data["points"] = points
        model_data["labels"] = labels
        model_data["surface"] = src
        
        return model_data, meta_info

    def get_shape_diff(self, index):
        pcl_path = self.split_lst_pcl_path[index]
        points_path = self.split_lst_points_path[index]
        depth_path = self.split_lst_depth_path[index] if self.use_depth else ""
        
        meta_info = {
            "pcl_file": pcl_path,
            "points_file": points_path,
            "depth_file": depth_path,
            "index": pcl_path.split('/')[-1].split('.')[0],
            "model": pcl_path.split('/')[-2]
        }
        
        pcl = np.load(pcl_path)
        surface = pcl['points']
        
        points = np.load(points_path)
        occ = np.unpackbits(points['occupancies'])
        points = points['points']
        

        if self.split.find('train') == -1: # val and test
            rng = np.random.default_rng(seed=self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
        else:
            rng = np.random.default_rng()
            
        if self.use_depth:
            depth = np.load(depth_path)
            depth = depth['points']
            src = depth
        else:
            src = surface
            
        surface_org = surface.copy()
        
        choice_inputs = np.random.choice(src.shape[0], self.n_inputs, replace=False)
        inputs = src[choice_inputs]
        
        if self.surface_sampling:
            ind = rng.choice(surface.shape[0], self.pc_size, replace=False)
            surface = surface[ind]
        
        noise = self.inputs_noise_std * np.random.randn(*inputs.shape)
        noise = noise.astype(np.float32)
        inputs = inputs + noise
        inputs = torch.from_numpy(inputs).float()
        surface = torch.from_numpy(surface).float()
        
        model_data = {}
        model_data["inputs"] = inputs
        model_data["surface"] = surface
        
        if not self.split == "train":
            surface_org = torch.from_numpy(surface_org).float()
            points = torch.from_numpy(points).float()
            occ = torch.from_numpy(occ).float()
            
            model_data["points"] = points
            model_data["labels"] = occ
            model_data["surface"] = surface_org
        
        return model_data, meta_info
    
    def get_deform_ae(self, index):
        # No evaluation implemented for this stage
        model = self.models[index]
        model_name = model["model"]
        model_indices = model["indices"]
        
        model_info = {}
        model_info["model"] = model_name
        model_info["indices"] = model_indices
        
        model_pcl_pathes = [os.path.join(self.dataset_path, model_name, "pcl_seq", f"{i:08d}.npz") for i in model_indices]
        model_near_pathes = [os.path.join(self.dataset_path, model_name, "points_near_seq", f"{i:08d}.npz") for i in model_indices]
        
        model_pcl_np = [np.load(model_pcl_path) for model_pcl_path in model_pcl_pathes]
        model_near_np = [np.load(model_near_path) for model_near_path in model_near_pathes]

        surface = np.array([model_pcl["points"] for model_pcl in model_pcl_np])
        near_points = np.array([model_near["points"] for model_near in model_near_np])

        ind = np.random.default_rng().choice(surface.shape[1], self.pc_size, replace=False)
        surface = surface[:,ind,:]

        surface_src = surface[0,:,:]
        surface_tgt = surface[1,:,:]

        surface_src = torch.from_numpy(surface_src)
        surface_tgt = torch.from_numpy(surface_tgt)

        ind = np.random.default_rng().choice(near_points.shape[1], self.num_samples, replace=False)# downsample to 1024
        near_points = near_points[:,ind,:]
        near_points_src = near_points[0,:,:]
        near_points_tgt = near_points[1,:,:]
        near_points_src = torch.from_numpy(near_points_src)
        near_points_tgt = torch.from_numpy(near_points_tgt)

        points_src = torch.cat([surface_src, near_points_src], dim=0)
        points_tgt = torch.cat([surface_tgt, near_points_tgt], dim=0)

        model_data = {}
        model_data["points_src"] = points_src.float()
        model_data["points_tgt"] = points_tgt.float()
        model_data["surface_src"] = surface_src.float()
        model_data["surface_tgt"] = surface_tgt.float()

        return model_data, model_info

    def get_deform_diff(self, index):
        model = self.models[index]
        model_info = {}
        
        model_name = model["model"]
        start_idx = model["start_idx"]
        
        model_info["model"] = model_name
        model_info["start_idx"] = start_idx
        model_data = {}

       
        end_idx = start_idx + self.length_sequence - 1 # [0, ..., 16] => 17 frames

        # Sample num_training_frames frames from the sequence
        if self.split == "train":
            sampled_idx = np.random.choice(range(start_idx, end_idx + 1), self.n_training_frames, replace=False)
        else:
            sampled_idx = range(start_idx, end_idx + 1)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Sort the sampled indices and preappend 0 to the list
        sampled_idx = sorted(sampled_idx)
        sampled_idx = [start_idx] + sampled_idx

        model_pcl_pathes = [os.path.join(self.dataset_path, model_name, "pcl_seq", f"{i:08d}.npz") for i in sampled_idx]
        model_shape_pcl_pathes = [os.path.join(self.dataset_path.replace("deform", "shape"), model_name, "pcl_seq", f"{i:08d}.npz") for i in sampled_idx]
        model_points_pathes = [os.path.join(self.dataset_path.replace("deform", "shape"), model_name, "points_seq", f"{i:08d}.npz") for i in sampled_idx]
        
        model_pcl_np = [np.load(model_pcl_path) for model_pcl_path in model_pcl_pathes]
        model_points_np = [np.load(model_points_path) for model_points_path in model_points_pathes]
        
        surface = np.array([model_pcl["points"] for model_pcl in model_pcl_np]) 
        points = np.array([model_points["points"] for model_points in model_points_np])
        occ = np.array([np.unpackbits(model_points["occupancies"]) for model_points in model_points_np])
        surface_org = surface.copy()
        points = torch.from_numpy(points)
        occ = torch.from_numpy(occ)
        if self.use_depth:
            model_depth_pathes = [os.path.join(self.dataset_path, model_name, f"depth_rotate45", f"{i:08d}.npz") for i in sampled_idx]
            model_depth_np = [np.load(model_depth_path) for model_depth_path in model_depth_pathes]
            src = np.array([model_depth["points"] for model_depth in model_depth_np])
            src = src.astype(np.float16)
        else:
            src = surface

        choice_inputs = np.random.choice(src.shape[1], self.n_inputs, replace=False)
        inputs = src[:, choice_inputs, :]
        noise = self.inputs_noise_std * np.random.randn(*inputs.shape)
        noise = noise.astype(np.float32)
        inputs = inputs + noise
        inputs = torch.from_numpy(inputs).float()
        
        if self.surface_sampling:
            if self.split.find('train') == -1:
                ind = np.random.default_rng(seed=self.seed).choice(surface.shape[1], self.pc_size, replace=False)
            else:
                ind = np.random.default_rng().choice(surface.shape[1], self.pc_size, replace=False)
            surface = surface[:,ind,:]

        surface = torch.from_numpy(surface)

        surface0 = surface[0,:,:]
        # repeat it to num_training_frames
        if self.split == "train":
            surface0 = surface0.repeat(self.n_training_frames, 1, 1)
        else:
            surface0 = surface0.repeat(self.length_sequence, 1, 1)
        surfaceT = surface[1:,:,:]
        
        inputs0 = inputs[0,:,:]
        # repeat it to num_training_frames
        if self.split == "train":
            inputs0 = inputs0.repeat(self.n_training_frames, 1, 1)
        else:
            inputs0 = inputs0.repeat(self.length_sequence, 1, 1)
            
        inputsT = inputs[1:,:,:]

        model_data["surface_src"] = surface0.float()
        model_data["surface_tgt"] = surfaceT.float()
        model_data["inputs_src"] = inputs0.float()
        model_data["inputs_tgt"] = inputsT.float()
        
        if not self.split == "train":
            surface_org = torch.from_numpy(surface_org).float()
            
            model_data["inputs"] = inputsT.float()
            
            # Surface must have corresponding
            
            model_data["surface"] = surface_org.float()[1:,:,:]
            model_data["points"] = points.float()[1:,:,:]
            model_data["labels"] = occ.float()[1:,:]
        
        return model_data, model_info