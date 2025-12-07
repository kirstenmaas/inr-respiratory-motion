from datetime import datetime
import wandb
import copy
import os

def config_parser(config_file='config.txt', sweep_file='sweep.yaml'):

    import configargparse
    import argparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, 
                        help='config file path', default=config_file, help='location of config file')
    parser.add_argument('--wandb_sweep_yaml', type=str, default=sweep_file, help='location of sweep file')
    parser.add_argument('--project_name', type=str, help='wandb project name')
    parser.add_argument('--nb_learning_frames', type=int, default=-1, choices=[-1, 15, 30, 45, 60], help='number of learning frames')

    parser.add_argument('--dataset', type=str, default='XCAV', choices=['XCAV'], help='dataset name')
    parser.add_argument('--patient_number', type=str, default='CVAI-1255', help='Dicom id')
    parser.add_argument('--dicom_number', type=int, default=0, help='Sequence id, in case patient has multiple sequences associated')
    
    parser.add_argument('--div_ratio', type=int, default=4, help='how dense we want to sample the image')
    parser.add_argument('--use_mask', default=False, action=argparse.BooleanOptionalAction, help='diaphragm mask: yes or no')

    parser.add_argument('--n_iterations_d', type=int, default=2500, help='number of learning iterations')
    parser.add_argument('--n_iterations_pf_i', type=int, default=2500, help='number of test-time iterations per frame')
    parser.add_argument('--save_every', type=int, default=500, help='save every x number of iterations')
    
    parser.add_argument('--lr_d', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_d_gamma', type=float, default=1e-4, help='exponential decay of lr')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')

    parser.add_argument('--in_features', type=int, default=2, help='number of regular MLP inputs')
    parser.add_argument('--hidden_channels_d', type=int, default=256, help='number of neurons')
    parser.add_argument('--num_layers_d', type=int, default=4, help='number of layers')

    parser.add_argument('--pos_enc_d', type=str, default='free', choices=['none', 'free', 'fourier'], help='type of positional encoding')
    parser.add_argument('--freq_basis_d', type=int, default=10, help='maximum frequency band')
    parser.add_argument('--freq_sigma_d', type=float, default=0., help='in case of fourier pos. enc: sigma')
    parser.add_argument('--freq_basis_d_s', type=int, default=0, help='start frequency band')
    parser.add_argument('--max_iterations_d', type=int, default=5000, help='number of iterations over which to do coarse-to-fine free encoding')

    parser.add_argument('--smooth_weight', type=float, default=1e-3, help='smooth weight for respiratory and contrast signals')
    parser.add_argument('--accel_weight', type=float, default=1e1, help='acceleration weight for contrast signal')
    parser.add_argument('--period_weight', type=float, default=1e-2, help='periodicity weight')
    parser.add_argument('--lipschitz_weight_d', type=float, default=1e-7, help='lipschitz regularization weight')

    return parser

def initialize_wandb(extra_chars=''):
    exp_name = datetime.now().strftime("%Y-%m-%d-%H%M%S") + extra_chars
    wandb.init(
        notes=exp_name,
    )
    return exp_name

def overwrite_args_wandb(run_args, wandb_args):
    # we want to overwrite the args based on the sweep args
    new_args = copy.deepcopy(run_args)
    for key in wandb_args.keys():
        setattr(new_args, key, wandb_args[key])
    
    return new_args

def initialize_save_folder(folder_name, exp_name):
    log_dir = folder_name + 'runs/' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir