from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import pickle


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_width, img_height, batch_size, n_cpu):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size_w=img_width, img_size_h=img_height,
                          augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_width

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(
                outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs,
                                               targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


# With augmentation
#python test.py --weights_path checkpoints_with_aug/yolov3_ckpt_19.pth 
# --model_def /datasets/datasets/bdd100k/yolo_files/without_gan/yolov3-bdd100k.cfg 
# --data_config /datasets/datasets/bdd100k/yolo_files/without_gan/bdd100k.data 
# --class_path /datasets/datasets/bdd100k/yolo_files/without_gan/bdd100k.names
# --batch_size=4 --pickle_file_name with_aug.pkl 0.9

# Without augmentation
#python test.py --weights_path checkpoints_without_aug/yolov3_ckpt_19.pth 
# --model_def /datasets/datasets/bdd100k/yolo_files/without_gan/yolov3-bdd100k.cfg 
# --data_config /datasets/datasets/bdd100k/yolo_files/without_gan/bdd100k.data 
# --class_path /datasets/datasets/bdd100k/yolo_files/without_gan/bdd100k.names
# --batch_size=4 --pickle_file_name without_aug.pkl --conf_thres 0.9

# With GAN
# python test.py --weights_path checkpoints_gan/yolov3_ckpt_19.pth --model_def ~/datasets/bdd100k/yolo_files/with_gan/yolov3-bdd100k.cfg --data_config ~/datasets/bdd100k/yolo_files/with_gan/bdd100k.data --class_path ~/datasets/bdd100k/yolo_files/with_gan/bdd100k.names --batch_size=4 --pickle_file_name with_gan.pkl --conf_thres 0.9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8,
                        help="size of each image batch")
    parser.add_argument("--model_def", type=str,
                        default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str,
                        default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str,
                        default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str,
                        default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5,
                        help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float,
                        default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size_w", type=int, default=800,
                        help="size of  wodth of each image dimension")
    parser.add_argument("--img_size_h", type=int, default=800,
                        help="size of height of each image dimension")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="id of GPU")
    parser.add_argument("--pickle_file_name", type=str,
                        default="data.pkl", help="pickle filename to save")
    opt = parser.parse_args()
    print(opt)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["test"]
    print(valid_path)
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_width=opt.img_size_w,
        img_height=opt.img_size_h,
        batch_size=opt.batch_size,
        n_cpu=opt.n_cpu
    )

    data = {'precision':precision,
            'recall': recall,
            'AP':AP,
            'f1':f1,
            'ap_class': ap_class}
    file_name = opt.pickle_file_name
    with open(file_name,'wb') as f:
        pickle.dump(data,f)

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"F1 score: {f1}")
