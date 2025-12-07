import neurokit2 as nk
import numpy as np
import struct as stru
import torch
import pdb

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch

        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return [self.ids[self.curr:self.curr+self.batch], self.curr]

def ecg_to_phase(ecg, sampling_rate):
    clean_ecg = nk.ecg_clean(ecg, sampling_rate)
    
    signals, _ = nk.ecg_peaks(clean_ecg, sampling_rate)
    rpeaks = signals["ECG_R_Peaks"]

    # signals = nk.ecg_phase(clean_ecg, rpeaks=rpeaks, sampling_rate=sampling_rate)
    
    phase_ecg = np.zeros_like(clean_ecg)
    ids = np.argwhere(rpeaks).astype('int')

    # fill start
    avg_length = np.mean(ids[1:] - ids[:-1])  

    start_id = int(ids[0])
    perc_start = start_id / avg_length
    phase_ecg[0:start_id+1] = np.linspace(1 - perc_start, 1, num=start_id+1)
    for i in range(len(ids)-1):
        id0 = int(ids[i])
        id1 = int(ids[i+1])
        phase_ecg[id0:id1+1] = np.linspace(0, 1, num=id1-id0+1)

    # fill end
    last_id = int(ids[-1])
    end_id = int(phase_ecg.shape[0])
    perc_end = (end_id - last_id) / avg_length
    phase_ecg[last_id:] = np.linspace(0, 1 - perc_end, num=end_id-last_id)

    return phase_ecg, ids

def get_ecg_from_dicom(dicom):

    # read curve data from dicom
    raw_ecg_signal = dicom[0x5000, 0x3000].value

    # DPPS = Data Points Per Second

    # unsigned short - data representation = 0
    numbers = np.array([stru.unpack('H', raw_ecg_signal[i:i+2])[0] for i in range(0, len(raw_ecg_signal), 2)])

    # frame time vector to determine the frame rate in msec
    # T(n) = sum_{i=0}^{n-1} t_i where t_i is in frame time vector to get relative time T(n) for frame n (DICOM C.7.6.5.1.2)
    frame_time_vector = dicom[0x0018, 0x1065].value
    start_time_per_frame = [np.sum(frame_time_vector[:i+1]) for i in range(len(frame_time_vector))]

    # cine rate === number of frames per second
    cine_rate = dicom[0x0018, 0x0040].value
    
    last_frame_start = start_time_per_frame[-1]
    total_time_ms = last_frame_start + frame_time_vector[-1]
    total_time_s = (total_time_ms) / 1e3

    sampling_rate = len(raw_ecg_signal) / total_time_s

    coordinate_start_value = int(dicom[0x5000, 0x0112].value)

    # not sure about how to define start and end time here? NOW: end of last frame is end time
    x_values = np.linspace(coordinate_start_value, total_time_ms, len(numbers)) #/ total_time_ms
    y_values = numbers

    return x_values, y_values, sampling_rate