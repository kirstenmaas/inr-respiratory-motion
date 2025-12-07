import matplotlib.pyplot as plt
import numpy as np

def correct_sign(predicted_signal, annotated_signal):
    corrected = False
    dot_product = np.dot(annotated_signal, predicted_signal)
    if dot_product < 0:
        corrected = True
        annotated_signal = -annotated_signal
    return annotated_signal, corrected

def norm_data(data):
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    return (data-mean_data)/(std_data)

def ncc(data0, data1):
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))
