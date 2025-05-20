from subprocess import call, Popen
import argparse
import os
import json
import time
import sys

parser = argparse.ArgumentParser(description='Run DySAT on all PIVEM datasets')

parser.add_argument('--min_time', type=int, default=2, help='min_time step')
parser.add_argument('--max_time', type=int, default=16, help='max_time step')
parser.add_argument('--run_parallel', type=str, default='False', help='Run in parallel')
parser.add_argument('--base_model', type=str, default='DySAT', help='Base model')
parser.add_argument('--model', type=str, default='default', help='Model variant')
parser.add_argument('--GPU_ID', type=int, default=0, help='GPU ID')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--featureless', type=str, default='True')
parser.add_argument('--max_gradient_norm', type=float, default=1.0)
parser.add_argument('--use_residual', type=str, default='False')
parser.add_argument('--neg_sample_size', type=int, default=10)
parser.add_argument('--walk_len', type=int, default=20)
parser.add_argument('--neg_weight', type=float, default=1.0)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--spatial_drop', type=float, default=0.1)
parser.add_argument('--temporal_drop', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--structural_head_config', type=str, default='16,8,8')
parser.add_argument('--structural_layer_config', type=str, default='128')
parser.add_argument('--temporal_head_config', type=str, default='16')
parser.add_argument('--temporal_layer_config', type=str, default='128')
parser.add_argument('--position_ffn', type=str, default='True')
parser.add_argument('--window', type=int, default=-1)

args = parser.parse_args()

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')

DATA_ROOT = "data"
train_file = "train.py" if args.base_model == "DySAT" else "train_incremental.py"

pivem_datasets = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])

for dataset_name in pivem_datasets:
    print("="*40)
    print("Running dataset:", dataset_name)

    # Create log dir
    output_dir = "./logs/" + args.base_model + "_" + args.model + "_" + dataset_name
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Update dataset name in args
    args.dataset = dataset_name

    # Save config
    with open(os.path.join(output_dir, 'flags_{}.json'.format(dataset_name)), 'w') as f:
        json.dump(vars(args), f)

    with open(os.path.join(output_dir, 'flags_{}.txt'.format(dataset_name)), 'w') as f:
        for k, v in vars(args).items():
            f.write("{}\t{}\n".format(k, v))

    # Build command list
    commands = []
    for t in range(args.min_time, args.max_time + 1):
        cmd = ' '.join([
            "python", train_file,
            "--time_steps", str(t),
            "--base_model", args.base_model,
            "--model", args.model,
            "--dataset", dataset_name
        ])
        commands.append(cmd)

    # Run
    if str2bool(args.run_parallel) and args.base_model == 'DySAT':
        print("Running in parallel on GPU", args.GPU_ID)
        procs = []
        for cmd in commands:
            print("Start:", cmd)
            proc = Popen(cmd, shell=True)
            time.sleep(10)
            procs.append(proc)
        for p in procs:
            p.wait()
    else:
        print("Running sequentially on GPU", args.GPU_ID)
        for cmd in commands:
            print("Call:", cmd)
            call(cmd, shell=True)

print("All PIVEM datasets processed.")
