from tqdm import tqdm
import torch
import wandb
import numpy as np
import time

import matplotlib.pyplot as plt

from networks.DensityMLP import DensityMLP
from objectives.regularizers import compute_periodic_loss, compute_smoothness_loss_resp, compute_smoothness_loss_contr, compute_second_order_smoothness_loss
from models.model_utils import correct_sign, ncc

class BackgroundModel():
    def __init__(self, args, dataset, log_dir, device):
        self.args = args
        self.dataset = dataset
        self.log_dir = log_dir
        self.device = device

        self.train_log = {}
        self.test_log = {}
    
    def init(self, args):
        self.setup_network(args)
        self.setup_optimizer(args)
        self.setup_criterion(args)
        self.setup_regularizers(args)

    def load_model(self, path):
        model_info = torch.load(f'{path}/background_model.pth', weights_only=True)
        self.network = DensityMLP(model_info['parameters'])
        self.network.to(self.device)
        self.network.load_state_dict(model_info['model'])
        self.network.eval()

    def setup_network(self, args):
        params = self.network_params(args)
        self.network = DensityMLP(params).to(self.device)
        
        self.anneal = (args.pos_enc_d == 'free' or args.pos_enc_d == 'anhash')
        self.max_iters_anneal = args.max_iterations_d if self.anneal else 0

    def network_params(self, args):
        return dict({
            "in_features": args.in_features,
            "hidden_channels": args.hidden_channels_d,
            "num_layers": args.num_layers_d,
            "pos_enc": args.pos_enc_d,
            "freq_basis": args.freq_basis_d,
            "freq_sigma": args.freq_sigma_d,
            "freq_gaussian": torch.randn([int(args.in_features) * int(args.freq_basis_d)]),
            "window_start": args.freq_basis_d_s,
            "device": self.device,
            "num_obsv": self.dataset.nr_frames,
            "frame_rate": self.dataset.frame_rate,
        })
    
    def setup_optimizer(self, args):
        latent_params = []
        network_params = []
        for name, param in self.network.named_parameters():
            if 'latent' or 'period' in name:
                latent_params.append(param)
            else:
                network_params.append(param)
        
        grad_vars = [
            {'params': list(network_params), 'lr': args.lr_d },
            {'params': list(latent_params), 'lr': args.lr_d }
        ]

        self.optimizer = torch.optim.Adam(grad_vars)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1 - args.lr_d_gamma)

    def setup_criterion(self, args):
        self.criterion = torch.nn.MSELoss().to(self.device)

    def setup_regularizers(self, args):
        if float(args.smooth_weight) > 0:
            self.apply_periodic_loss = True
            self.ecg_period = self.dataset.ecg_period
            self.periodic_loss_weight = float(args.period_weight)

            self.apply_smoothness_loss = True
            self.smoothness_weight = float(args.smooth_weight)
            self.second_order_smoothness_weight = float(args.accel_weight)
        
        self.apply_lipschitz_loss = False
        if float(args.lipschitz_weight_d) > 0:
            self.apply_lipschitz_loss = True
            self.lipschitz_weight = float(args.lipschitz_weight_d)

    def annealing_step(self, n_iter):
        if self.anneal:
            self.network.encoder.update_alpha(n_iter, max_iter=self.max_iters_anneal)
            self.train_log['window'] = self.network.encoder.ptr

    def train_step(self, args, n_iter, infer_frame_id=-1):
        self.network.train()

        min_frame_id = 0
        max_frame_id = int(self.dataset.nr_frames)
        resp_weights = torch.ones(int(self.dataset.nr_frames))
        if args.nb_learning_frames > -1:
            max_frame_id = int(args.nb_learning_frames)
            resp_weights = torch.cat((
                torch.ones(int(args.nb_learning_frames)),
                torch.zeros(int(self.dataset.nr_frames - args.nb_learning_frames))
            ))

        max_frame_id = int(args.nb_learning_frames) if args.nb_learning_frames > -1 else int(self.dataset.nr_frames)
        if infer_frame_id > -1:
            min_frame_id = infer_frame_id
            max_frame_id = infer_frame_id + 1

            nb_learnable_frames = self.dataset.nr_frames - args.nb_learning_frames

            # weight of the learnable frames is determined based on how far away we are (how many new frames did we introduce?)
            resp_weights = torch.ones(int(self.dataset.nr_frames))
            learn_frame_weights = resp_weights[:args.nb_learning_frames] - ((min_frame_id - args.nb_learning_frames + 1) / (nb_learnable_frames))
            resp_weights[:args.nb_learning_frames] = learn_frame_weights

            # new frames should not be taken into account for learning
            resp_weights[max_frame_id:] = 0.

        resp_weights = torch.clamp(resp_weights, min=1e-8, max=1. - 1e-8).to(self.device)

        next_ids, _ = self.dataset.training_sampler.nextids()
        coords = self.dataset.train_coords[next_ids]
        frame_ids = torch.randint(low=min_frame_id, high=int(max_frame_id), size=(next_ids.shape[0],), device=self.device)
        gt_densities = self.dataset.train_frames[frame_ids, next_ids]

        pred_densities = self.network(coords, frame_ids)

        pixel_loss = self.criterion(pred_densities, gt_densities)
        total_loss = pixel_loss
        self.train_log['pixel_loss'] = pixel_loss.item()

        if self.apply_periodic_loss:
            periodic_weight = self.periodic_loss_weight
            
            if self.dataset.ecg_period is None: 
                ecg_period = torch.nn.functional.relu(self.network.ecg_period)
                ecg_period_frames = (self.network.max_hr_frames - self.network.min_hr_frames) * ecg_period + self.network.min_hr_frames
            else:
                ecg_period_frames = torch.tensor([self.dataset.ecg_period], device=self.device).float()

            periodic_loss = compute_periodic_loss(self.network.cardiac_latent, ecg_period_frames)
            total_loss += periodic_weight * periodic_loss
            self.train_log['periodic_loss'] = periodic_loss.item()
            self.train_log['learned_ecg_frames'] = ecg_period_frames

        if self.apply_smoothness_loss:
            self.smoothness_weight = self.args.smooth_weight

            smoothness_loss_resp = compute_smoothness_loss_resp(self.network.respiratory_latent, resp_weights)
            smoothness_loss_contr = compute_smoothness_loss_contr(self.network.contrast_latent)
            
            total_loss += self.smoothness_weight * smoothness_loss_resp + self.smoothness_weight * smoothness_loss_contr
            self.train_log['smoothness_loss_resp'] = smoothness_loss_resp.item()
            self.train_log['smoothness_loss_contr'] = smoothness_loss_contr.item()

            second_order_smoothness_loss = compute_second_order_smoothness_loss(self.network.contrast_latent)
            total_loss += self.second_order_smoothness_weight * second_order_smoothness_loss
            self.train_log['second_order_smoothness_loss'] = second_order_smoothness_loss.item()

        if self.apply_lipschitz_loss:
            lipschitz_loss = self.network.get_lipschitz_loss()
            total_loss += self.lipschitz_weight * lipschitz_loss
            self.train_log['lipschitz_loss_d'] = lipschitz_loss.item()

        self.train_log['total_loss'] = total_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        self.train_log['lr_d'] = self.optimizer.param_groups[0]['lr']

    def test_step(self, args, n_iter, log_every=10, inference_step=False):
        self.network.eval()

        with torch.no_grad():
            coords = self.dataset.test_coords
            for i in range(0, self.dataset.nr_frames, log_every):
                gt_densities = self.dataset.test_frames[i]
                blatent = torch.Tensor([i]).repeat(coords.shape[0]).to(self.device).int()
                pred_densities = self.network(coords, blatent).reshape(gt_densities.shape)

                pixel_loss = self.criterion(pred_densities, gt_densities)
                self.test_log[f'test_pixel_loss_{i}'] = pixel_loss.item()
                self.log_test_images(gt_densities, pred_densities, i)
            
            self.test_latent_log(n_iter, args, inference_step=inference_step)

            self.network.save(f"{self.log_dir}/background_model.pth", {})

    def log_test_image(self, img):
        np_img = img.cpu().numpy()
        np_img = (np_img - np.min(np_img)) / (np.max(np_img) - np.min(np_img))
        return np_img

    def log_test_images(self, gt, pred, image_idx):
        np_gt = self.log_test_image(gt)
        np_pred = self.log_test_image(pred)
        np_diff = np.abs(np_gt - np_pred)

        self.test_log[f'gt-{image_idx}'] = wandb.Image(np_gt)
        self.test_log[f'pred-{image_idx}'] = wandb.Image(np_pred)
        self.test_log[f'diff-{image_idx}'] = wandb.Image(np_diff)

    def test_latent_log(self, n_iter, args, lst_names=['cardiac', 'respiratory', 'contrast'], inference_step=False):
        latents = torch.cat((self.network.cardiac_latent, self.network.respiratory_latent, self.network.contrast_latent), dim=-1)
        latents = latents.cpu().numpy()
        for i in range(latents.shape[1]):
            x_values = np.arange(0, latents.shape[0])
            y_values = latents[:,i]
            plt.plot(x_values, y_values, label=lst_names[i])

            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns=["x", "y"])
            self.test_log[f"Latent-{lst_names[i]}-{n_iter}"] = wandb.plot.line(table, "x", "y", title=f"Latent-{lst_names[i]}-{n_iter}")            

        if self.dataset.add_data is not None and 'annotated' in self.dataset.add_data.keys():
            respiratory_signal = latents[:,1]
            annotated_signal = self.dataset.add_data['annotated']

            # plot annotated signal
            x_values = np.arange(0, latents.shape[0])
            y_values = annotated_signal
            plt.plot(x_values, y_values, label=lst_names[i])
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns=["x", "y"])
            self.test_log[f"Annotated-Signal"] = wandb.plot.line(table, "x", "y", title=f"Annotated-Signal")            

            annotated_signal, corrected = correct_sign(respiratory_signal, annotated_signal)
            ncc_value = ncc(annotated_signal, respiratory_signal)
            self.test_log['NCC_annotated_respiratory'] = ncc_value

        plt.legend()
        plt.savefig(f'{self.log_dir}/latents-{inference_step}-{n_iter}.png')
        plt.close()

    def fit(self):
        start_training_time = time.time()
        pbar = tqdm(range(self.args.n_iterations_d+1), miniters=10)

        # save_every = self.args.save_every if self.args.save_every > -1 else self.args.n_iterations_d

        for n_iter in pbar:
            log_dict = {}

            self.annealing_step(n_iter)
            self.train_step(self.args, n_iter)
            pbar.set_description(f"Iter {n_iter} | Pixel loss: {self.train_log['pixel_loss']:.4f}")
            log_dict.update(self.train_log)

            # if n_iter % save_every == 0:
            #     self.test_step(self.args, n_iter)
            #     log_dict.update(self.test_log)
            
            wandb.log(log_dict)
        log_dict.update({ 'training_time': time.time() - start_training_time })
        self.test_step(self.args, n_iter, inference_step=True)
        log_dict.update(self.test_log)
        wandb.log(log_dict)

    def infer(self):
        # freeze the network weights, not the latents
        self.network.freeze_weights()

        # reset learning rate for latent optimization
        self.optimizer.param_groups[1]['lr'] = self.args.lr_d

        infer_frame_ids = np.arange(self.args.nb_learning_frames, self.dataset.nr_frames)

        # save_every = self.args.save_every if self.args.save_every > -1 else self.args.n_iterations_d
        
        start_infer_time = time.time()
        for infer_frame_id in infer_frame_ids:
            for n_iter in range(self.args.n_iterations_pf_i+1):
                log_dict = {}

                self.train_step(self.args, n_iter, infer_frame_id=infer_frame_id)
                log_dict.update(self.train_log)

                # if n_iter % save_every == 0:
                    # self.test_step(self.args, n_iter, inference_step=True)
                    # log_dict.update(self.test_log) 
                
                wandb.log(log_dict)
        inference_time = time.time() - start_infer_time
        self.test_step(self.args, n_iter, inference_step=True)
        log_dict.update(self.test_log)
        log_dict.update({ 'inference_time': inference_time / len(infer_frame_ids) })
        
        wandb.log(log_dict)

    def get_ref_signal(self, signal):
        signal = ( signal - torch.min(signal)) / \
            (torch.max(signal) - torch.min(signal))
        signal = signal - signal[self.dataset.ref_frame_id]
        return signal

    def set_learned_latents(self):
        self.network.freeze()
        with torch.no_grad():
            saved_latents = torch.cat((self.network.cardiac_latent, self.network.respiratory_latent, self.network.contrast_latent), dim=-1)

            cardiac_signal = self.get_ref_signal(saved_latents[:,0])

            respiratory_signal = saved_latents[:,1]
            self.dataset.store_respiratory_signal(respiratory_signal)
            respiratory_signal = self.get_ref_signal(respiratory_signal).float()

            self.saved_latents = saved_latents

            contrast_signal = self.get_ref_signal(saved_latents[:,2])
            self.dataset.set_learned_latents(cardiac_signal, respiratory_signal, contrast_signal)
