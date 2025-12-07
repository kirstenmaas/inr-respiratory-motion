import numpy as np
import matplotlib.pyplot as plt

from pca_utils import normalize_unit_norm
from data_utils import read_data, correct_sign, plot_sota, get_annotations

def annotate_video(video):
    T, H, W = video.shape
    points = [None] * T
    fig, ax = plt.subplots(figsize=(15, 15))

    state = { 't': 0 }

    def show_frame():
        ax.clear()
        ax.imshow(video[state['t']], cmap='gray', origin='upper')
        ax.set_title(f"Frame {state['t']+1}/{T}")

        # Show previous point if available
        if state['t'] > 0 and points[state['t']-1] is not None:
            prev_x, prev_y = points[state['t']-1]
            ax.plot(prev_x, prev_y, 'ro', markersize=4, alpha=0.1, label='Previous point')
            ax.legend()

        state['hover_artist'], = ax.plot([], [], 'gx', markersize=16, label='Hover')
        plt.draw()

    def on_move(event):
        if event.inaxes != ax:
            return
        if state['hover_artist'] is not None:
            state['hover_artist'].set_data(event.xdata, event.ydata)
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        points[state['t']] = (event.xdata, event.ydata)
        state['t'] += 1
        if state['t'] >= T:
            plt.close(fig)
        else:
            show_frame()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    show_frame()
    plt.show()

    return np.array(points)

if __name__ == "__main__":
    # code to annotate position of diaphragm with a single point

    dataset = 'XCAV'
    patient_number = 'CVAI-1255'
    dicom_id = 'CVAI-1255LAO52_CAU1'
    annotate = True

    store_path = f'D:/respiratory_motion_data/{dataset}/{patient_number}/{dicom_id}/'
    sequence = read_data(dicom_id, patient_number, dataset)

    if annotate:
        points = annotate_video(sequence)

        distances = points[0][1] - points[:, 1]
        annotated_signal = normalize_unit_norm(distances)
        annotated_signal, _ = correct_sign(np.loadtxt(f'{store_path}/pca.txt'), annotated_signal)
    
    curr_annotations, counter = get_annotations(store_path)

    if annotate:
        np.savetxt(f'{store_path}/annotated-{counter}.txt', annotated_signal)

        if counter > 0:
            curr_annotations = np.concatenate((curr_annotations.reshape((sequence.shape[0], -1)), 
                                            annotated_signal.reshape((sequence.shape[0], -1))), axis=-1)
        else:
            curr_annotations = annotated_signal.reshape((sequence.shape[0], -1))

    for idx, ann in enumerate(curr_annotations.T):
        plt.plot(np.arange(0, len(ann)), ann, label=f'annotation-{idx}')
    plt.legend()
    plt.savefig(f'{store_path}/annotations.png')
    plt.close()

    if counter > 0:
        mean_annotation = np.mean(curr_annotations, axis=-1)
        np.savetxt(f'{store_path}/annotated.txt', mean_annotation)
    elif annotate:
        np.savetxt(f'{store_path}/annotated.txt', annotated_signal)

    annotated = np.loadtxt(f'{store_path}/annotated.txt')

    plot_sota(annotated, store_path)