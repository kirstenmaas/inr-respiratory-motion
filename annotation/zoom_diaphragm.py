import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from data_utils import read_data

def annotate_video(video):
    T, H, W = video.shape
    areas = [None] * T
    fig, ax = plt.subplots(figsize=(15, 15))

    state = { 't': 0, 'selector': None, 'rect_coords': None }

    def show_frame():
        ax.clear()
        ax.imshow(video[state['t']], cmap='gray', origin='upper')
        ax.set_title(f"Frame {state['t']+1}/{T}")

        # redraw previous annotation (if any)
        if areas[state['t']] is not None:
            x1, y1, x2, y2 = areas[state['t']]
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor='red', linewidth=2))

        fig.canvas.draw_idle()

    def onselect(eclick, erelease):
        state['rect_coords'] = (np.around(eclick.xdata), np.around(eclick.ydata),
                                np.around(erelease.xdata), np.around(erelease.ydata))

    def on_key(event):
        if event.key == 'enter' and state['rect_coords'] is not None:
            areas[state['t']] = state['rect_coords']
            state['t'] += 1
            print(state['t'])
            if state['t'] >= T:
                plt.close(fig)
            else:
                show_frame()

    state['selector'] = RectangleSelector(ax, onselect,
                                          useblit=True, button=[1],
                                          minspanx=5, minspany=5,
                                          interactive=True)

    fig.canvas.mpl_connect('key_press_event', on_key)

    show_frame()
    plt.show()

    return np.array(areas)

if __name__ == "__main__":
    # code to annotate the zooming "rectangle" for the diaphragm visiblity experiments

    dataset = 'XCAV'
    patient_number = 'CVAI-1255'
    dicom_id = 'CVAI-1255LAO52_CAU1'

    store_path = f'D:/respiratory_motion_data/{dataset}/{patient_number}/{dicom_id}/'
    sequence = read_data(dicom_id, patient_number, dataset)

    areas = annotate_video(sequence)
    non_empty_areas = np.array([area for area in areas if area.any()])
    if len(areas) == len(non_empty_areas):
        # Save raw rectangles (x1, y1, x2, y2 per frame)
        np.savetxt(f'{store_path}/areas.txt', areas, fmt="%.4f")