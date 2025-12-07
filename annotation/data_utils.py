import numpy as np
import matplotlib.pyplot as plt
import os

def read_data(dicom_id, patient_number, dataset):
    if dataset == 'XCAV':
        data_path = f'D:/angio_datasets/XCAV/{patient_number}/images/{dicom_id}/'
        sequence = []
        for file in os.listdir(data_path):
            sequence.append(plt.imread(f'{data_path}/{file}'))
        sequence = np.array(sequence)
    return sequence

def correct_sign(pca_signal, annotated_signal):
    corrected = False
    dot_product = np.dot(annotated_signal, pca_signal)
    if dot_product < 0:
        corrected = True
        annotated_signal = -annotated_signal
    return annotated_signal, corrected

def plot_sota(annotated, store_path, options=['own', 'pca', 'kpca', 'multi_res_kpca']):
    annotated_n = (annotated - np.min(annotated)) / (np.max(annotated) - np.min(annotated))

    num_signals = len(options)
    fig, axes = plt.subplots(num_signals, 1, sharex=True, figsize=(6, 2*num_signals))

    for idx, option in enumerate(options):
        ax = axes[idx]

        try:
            signal = np.loadtxt(f'{store_path}/{option}.txt')
            
            signal, corrected = correct_sign(annotated, signal)
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

            ax.plot(np.arange(len(signal)), signal, label=option, color='blue')
            ax.plot(np.arange(len(annotated)), annotated_n, linestyle='--', color='black', label='annotation')
        except:
            print(f'No {option} found')
        ax.set_ylabel("Signal")
        ax.legend(loc='upper right')
    
    tick_step = 10
    axes[-1].set_xticks(np.arange(0, len(annotated) + 1, tick_step))
    axes[-1].set_xlabel("Frames")
    plt.legend()
    plt.savefig(f'{store_path}/all.png')
    plt.close()

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def get_annotations(folder):
    counter = 0
    annotations = []
    for fname in os.listdir(folder):
        if 'annotated' in fname and has_numbers(fname) and '.txt' in fname:
            print(fname)
            curr_ann = np.loadtxt(f'{folder}/{fname}')
            annotations.append(curr_ann)
            counter += 1
    return np.array(annotations).T, counter

def norm_data(data):
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    return (data-mean_data)/(std_data)

def ncc(data0, data1):
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))