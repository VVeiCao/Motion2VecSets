import os
import argparse
import yaml
from munch import Munch
import torch

# Get available GPUs
GPU_COUNT = torch.cuda.device_count()
if GPU_COUNT < 1:
    raise ValueError('No GPU available')
    
parser = argparse.ArgumentParser(description='Run a command')
parser.add_argument('--config_path', type=str, help='config path', default=None, required=True)
parser.add_argument('--eval', action='store_true', help='run eval')
parser.add_argument('--test_ui', action='store_true', help='run test_ui')
args = parser.parse_args()

config = Munch(yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader))

# print(config.lr)
run_cmd = f"torchrun \
        --nproc_per_node={GPU_COUNT} core/{config.stage}/{config.mode}/main_{config.mode}_{config.stage}.py \
        --accum_iter={config.accum_iter} \
        --save_every {config.save_every} \
        --eval_every {config.eval_every} \
        --lr {config.lr} \
        --batch_size {config.batch_size} \
        --epochs {config.epochs} \
        --warmup_epochs {config.warmup_epochs} \
        --config_path {args.config_path} \
        "

if config.mode == 'ae':
    run_cmd += f"--kl_weight {config.kl_weight}"
        
if config.resume:
    run_cmd += f" --resume {config.resume}"
    
if args.eval:
    test_tag = "--test_ui" if args.test_ui else ""
    run_cmd += f" --eval {test_tag}"  

os.system(run_cmd)