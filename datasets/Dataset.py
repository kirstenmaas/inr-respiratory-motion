import pandas as pd
import pydicom
import numpy as np
import torch
import pdb
import os

from skimage.transform import resize

import matplotlib.pyplot as plt

from datasets.data_utils import SimpleSampler

class Dataset():
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.load_data(args)
        self.set_train_coords(args)
        self.set_training_sampler(args)
        self.set_test_coords(args)
        self.load_add_signals()

    def load_xcav(self, data, args, set_ref_id, store_path, data_path='D:/angio_datasets/XCAV'):
        row = data.iloc[args.dicom_number]
        dicom_id = row['Dicom']
        print(f'Loading {dicom_id} of patient {args.patient_number} of dataset {args.dataset}...')

        ref_frame_id = int(row['Ref_frame'])
        ref_frame_id_2 = int(row['Ref_frame']) + 1

        self.dicom = None
        self.dicom_id = dicom_id
        self.store_folder_name = f'{store_path}/{args.dataset}/{args.patient_number}/{self.dicom_id}'

        self.ref_frame_ids = [ref_frame_id, ref_frame_id_2]
        self.ref_frame_id = self.ref_frame_ids[set_ref_id]

        sequence_folder = f'{data_path}/{args.patient_number}/images/{dicom_id}/'
        self.load_image_sequence(sequence_folder, frame_rate=12.5)

        self.ecg_period = None

    def load_data(self, args, set_ref_id=0, store_path='datasets/'):
        # assumes there is a csv file that specifies the ddataset, patients, and dicom ids
        data = pd.read_csv('data.csv', delimiter=';')
        data = data[data['Dataset'] == args.dataset]
        data = data[data['Patient'] == args.patient_number]
        
        if args.dataset == 'XCAV':
            self.load_xcav(data, args, set_ref_id, store_path=store_path)
        return

    def load_image_sequence(self, sequence_folder, frame_rate=15, start_frame_id=0):
        sequence = []
        for file in os.listdir(sequence_folder):
            sequence.append(plt.imread(f'{sequence_folder}/{file}'))
        sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))
        sequence = sequence[start_frame_id:]

        if self.args.use_mask:
            areas = np.loadtxt(f'{self.store_folder_name}/areas.txt')
            area = areas[0].astype('int')
            sequence = sequence[:, area[1]:area[3], area[0]:area[2]]

        self.nr_frames, self.height, self.width = sequence.shape

        self.frame_rate = float(frame_rate)
        ref_frame = torch.from_numpy(sequence[self.ref_frame_id]).to(self.device).float()

        self.sequence = sequence
        self.ref_frame = ref_frame
        self.filtered_ref_frame = None
        self.res_ref_frame = None

    def set_train_coords(self, args):
        div_ratio = args.div_ratio
        
        self.ds_sequence = resize(self.sequence, (self.nr_frames, self.height // div_ratio, self.width // div_ratio))
        _, self.ds_height, self.ds_width = self.ds_sequence.shape
        self.train_frames = torch.from_numpy(self.ds_sequence).to(self.device).float().reshape(self.nr_frames, -1, 1)

        self.train_coords = torch.from_numpy(np.stack(np.meshgrid(
            np.linspace(-1., 1, self.ds_width, endpoint=False), 
            np.linspace(-1., 1, self.ds_height, endpoint=False)), 
        -1).reshape(-1, 2)).to(self.device).float()

    def set_training_sampler(self, args):
        self.training_sampler = SimpleSampler(self.ds_height*self.ds_width, args.batch_size)

    def set_test_coords(self, args):
        self.test_frames = torch.from_numpy(self.sequence).to(self.device).float()
        self.test_coords = torch.from_numpy(np.stack(np.meshgrid(
            np.linspace(-1., 1, self.width, endpoint=False), 
            np.linspace(-1., 1, self.height, endpoint=False)),
        -1)).reshape(-1, 2).to(self.device).float()
    
    def set_learned_latents(self, cardiac_signal, respiratory_signal, contrast_signal):
        self.cardiac_latent = cardiac_signal
        self.respiratory_latent = respiratory_signal
        self.contrast_latent = contrast_signal
    
    def load_add_signals(self):
        add_data = {
            'pca': np.loadtxt(f'{self.store_folder_name}/pca.txt'),
            'multi_res_kpca': np.loadtxt(f'{self.store_folder_name}/multi_res_kpca.txt'),
        }

        if os.path.isfile(f'{self.store_folder_name}/annotated.txt'):
            add_data['annotated'] = np.loadtxt(f'{self.store_folder_name}/annotated.txt')

        self.add_data = add_data

    def store_respiratory_signal(self, respiratory_signal):
        add_data = self.add_data

        respiratory_signal = respiratory_signal.cpu().numpy()
        
        resp_c = respiratory_signal - np.mean(respiratory_signal)
        norm = np.linalg.norm(resp_c)
        resp_norm = resp_c / (norm + 1e-12)

        # solve sign ambiguity
        dot_product = np.dot(resp_norm, add_data['pca'])
        if dot_product < 0:
            resp_norm = -resp_norm
        
        store_string = 'own'
        if self.args.nb_learning_frames > -1:
            store_string = f'{store_string}-{self.args.nb_learning_frames}'
        if self.args.use_mask:
            store_string = f'{store_string}-mask'
        np.savetxt(f'{self.store_folder_name}/{store_string}.txt', resp_norm)