import torch
import matplotlib.pyplot as plt
import wandb
import yaml
import json

from datasets.Dataset import Dataset
from models.BackgroundModel import BackgroundModel

from train_utils import config_parser, initialize_wandb, overwrite_args_wandb, initialize_save_folder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.switch_backend('agg')

def main():
    train()

def train():
    parser = config_parser()
    run_args = parser.parse_args()
    run_args = overwrite_args_wandb(run_args, wandb.config)
    wandb.log(vars(run_args))
    exp_name = initialize_wandb(f'-{run_args.project_name}')

    store_folder_name = f'cases/{run_args.patient_number}/'
    log_dir = initialize_save_folder(store_folder_name, exp_name)

    if run_args.config is not None:
        f = f'{log_dir}config.json'
        with open(f, 'w') as file:
            file.write(json.dumps(vars(run_args)))

    dataset = Dataset(run_args, device)
    if not dataset.dicom_id:
        return Exception('No such dicom id for this patient')

    background_model = BackgroundModel(run_args, dataset, log_dir, device)

    background_model_path = f'{dataset.store_folder_name}/background_model.pth'
    print('Training Background Model...')
    background_model.init(run_args)

    background_model.fit()
    background_model.network.save(f'{log_dir}/background_model.pth', {})

    background_model_path = f'{dataset.store_folder_name}/background_model'
    if (run_args.nb_learning_frames > -1):
        background_model.infer()
        background_model_path = f'{background_model_path}-{run_args.nb_learning_frames}'
    if (run_args.use_mask):
        background_model_path = f'{dataset.background_model_path}-mask'
    background_model_path = f'{background_model_path}.pth'
    # background_model.network.save(background_model_path, {}) # overwrite global model
    background_model.network.save(f'{log_dir}/background_model.pth', {}) # save local model

    print('Storing Learned Images and Latents...')
    background_model.set_learned_latents()

if __name__ == "__main__":

    parser = config_parser()
    run_args = parser.parse_args()

    wandb.login()

    project_name = run_args.project_name
    with open(run_args.wandb_sweep_yaml, 'r') as f:
        sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wandb.agent(sweep_id, function=main)