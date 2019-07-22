import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt


def show_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'scalars': 100
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print('All tags:')
    print(event_acc.Tags())

    val_f1 = event_acc.Scalars('val_f1')
    val_precision = event_acc.Scalars('val_precision')
    val_mAP = event_acc.Scalars('val_mAP')
    val_recall = event_acc.Scalars('val_recall')

    print(val_f1)

    steps = 100
    x = np.arange(steps)
    y = np.zeros([steps, 4])

    for i in range(steps):
        y[i, 0] = val_f1[i][2]
        y[i, 1] = val_precision[i][2]
        y[i, 2] = val_mAP[i][2]
        y[i, 3] = val_recall[i][2]

    plt.plot(x, y[:, 0], label='val_f1')
    plt.plot(x, y[:, 1], label='val_precision')
    plt.plot(x, y[:, 2], label='val_mAP')
    plt.plot(x, y[:, 3], label='val_recall')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Validation results")
    plt.legend(loc='upper right', frameon=True)
    plt.savefig('validation_results.png')


if __name__ == '__main__':
    log_file = "./PyTorch-YOLOv3/logs/events.out.tfevents.1562327706.af1c1de51c5a"
    show_tensorflow_log(log_file)
