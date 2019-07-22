import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


def show_tensorflow_log(path, title_text='', output_image_name='validation_results.png'):

    # Loading too much data is slow...
    tf_size_guidance = {
        'scalars': 100
    }

    print('Start data accumulation')
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

    for i in tqdm(range(steps)):
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
    plt.title(title_text)
    plt.legend(loc='upper right', frameon=True)
    plt.savefig(output_image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str,
                        help="path to events file")
    parser.add_argument("--title_text", type=str, default='',
                        help="title of output plot")
    parser.add_argument("--output_image_name", type=str, default='validation_results.png',
                        help="title of output plot")
    opt = parser.parse_args()
    log_file = opt.log_file
    title_text = opt.title_text
    output_image_name = opt.output_image_name
    show_tensorflow_log(log_file, title_text=title_text,
                        output_image_name=output_image_name)
