import os
import csv
import time
import torch
import logging
import subprocess
import numpy as np
import torch.nn as nn
from collections import defaultdict
from config import DataParameters as dp
from typing import List, NoReturn, Dict, Union


def create_debug_directories() -> NoReturn:
    # Create the debug directory, if it does not already exist
    if not os.path.exists('./debug'):
        os.mkdir('./debug')

    # Create the logs, weights, and summary directories inside debug, if they do not already exist
    if not os.path.exists('./debug/logs'):
        os.makedirs('./debug/logs')
    if not os.path.exists('./debug/weights'):
        os.makedirs('./debug/weights')
    if not os.path.exists('./debug/summary'):
        os.makedirs('./debug/summary')


def remove_values(src_list: List[str], del_list: List[str]) -> List[str]:
    """
    Remove values from src_list that are in del_list.
    :param src_list: List of strings.
    :param del_list: List of strings to remove from src_list.
    :return: New list with values from src_list that are not in del_list.
    """
    return [val for val in src_list if val not in del_list]


def create_csv_file(path: str, filename: str) -> None:
    """
    Create a CSV file at the given path and filename, containing information from image and annotation files.
    :param path: Path to directory containing image and annotation files.
    :param filename: Name of CSV file to create.
    """
    img_path = os.path.join(path, 'images')
    ann_path = os.path.join(path, 'annotations')

    img_list = os.listdir(img_path)
    ann_list = os.listdir(ann_path)

    files_pr = ['_aro.npy', '_val.npy', '_exp.npy']

    with open(os.path.join(path, f'{filename}.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        csv_writer.writerow(['Image Path', 'Arousal', 'Valence', 'Emotion'])

        for img in img_list:
            img_num = os.path.splitext(img)[0]
            atr_ = [img_num + fl_pr for fl_pr in files_pr]

            if all(a in ann_list for a in atr_):
                arousal = float(np.load(os.path.join(ann_path, atr_[0])))
                valence = float(np.load(os.path.join(ann_path, atr_[1])))
                emotion = float(np.load(os.path.join(ann_path, atr_[2])))

                ann_list = remove_values(ann_list, atr_)

                csv_writer.writerow([os.path.join(img_path, img), arousal, valence, emotion])

    print(f'CSV file was generated with {len(img_list)} images')


def get_dataset(path_to_csv_file: str, label: str) -> Dict[str, List[Union[str, int, float]]]:
    """
    Loads data from a CSV file into a dictionary containing lists of images and their corresponding labels.
    :param path_to_csv_file: The path to the CSV file.
    :param label: Label to use to create a list of labels (arousal, valence, emotions).
    :return:Dictionary with keys 'images' and 'labels', containing lists of names images and labels.
    """
    valid_labels = ['arousal', 'valence', 'emotions']
    if label not in valid_labels:
        raise ValueError(f'Invalid label value. Choose from {valid_labels}')

    images, labels = [], []
    with open(path_to_csv_file, mode='r', encoding='utf-8-sig') as file:
        csv_file = csv.reader(file, delimiter=';')
        for line in csv_file:
            images.append(str(line[0]))
            if label == 'arousal':
                labels.append(float(line[1]))
            elif label == 'valence':
                labels.append(float(line[2]))
            elif label == 'emotions':
                labels.append(int(line[3]))

    data_dict = {'images': images, 'labels': labels}
    return data_dict


def get_loss_func(label: str, class_weights: List[float | int] = None) -> nn.Module:
    """
    Return the appropriate loss function based on the label.
    :param class_weights: weights of imbalanced dataset
    :param label: A string indicating the type of label (emotions, arousal, or valence).
    :return: An instance of a PyTorch loss function module.
    """
    loss_criterion = {'emotions': nn.CrossEntropyLoss(weight=class_weights),
                      'arousal': nn.MSELoss(),
                      'valence': nn.MSELoss()
                      }

    if label not in loss_criterion:
        raise ValueError(f"Invalid label type '{label}', must be 'emotions', 'arousal' or 'valence'.")

    return loss_criterion[label]


def get_optimizer(model: torch.nn.Module, label: str,
                  learning_rate: float = 1e-3, weight_decay: float = 0) -> torch.optim.Optimizer:
    """ Returns the model optimizer for the given label """
    optimizer_dict = {'emotions': torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
                      'arousal': torch.optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
                      'valence': torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                      }

    if label not in optimizer_dict:
        raise ValueError(f"Invalid label type '{label}', must be 'emotions', 'arousal' or 'valence'.")

    return optimizer_dict[label]


def count_class_weights(labels_list: List[int]) -> List[float | int]:
    """Return weights classes for imbalanced dataset"""
    counts = defaultdict(int)

    for label in labels_list:
        counts[label] += 1

    class_counts = [counts[label] for label in sorted(counts.keys())]
    return 1.0 / torch.tensor(class_counts, dtype=torch.float)


def compare_video(vid_1: str = None, vid_2: str = None, vid_3: str = None, vid_out: str = None) -> None:

    bash_command = [f'ffmpeg -i {vid_1} -i {vid_2} -i {vid_3} -filter_complex ' +
                    f'[0:v:0][1:v:0][2:v:0]hstack=inputs=3 -c:v libx264 -tune film -crf 16 -b:a 256k {vid_out}']

    for command in bash_command:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        print('Video successfully merged')

    time.sleep(3)
    for file in os.listdir('./data/output_video'):
        if file != 'result.mp4':
            os.remove(os.path.join(dp.output_video, file))

    for file in os.listdir('./data/input_video'):
        os.remove(os.path.join(dp.input_video, file))


def create_logger(log_dir: str, filename: str):
    """ """
    log_file = os.path.join(log_dir, f'log_{filename}.txt')
    log = logging.getLogger(filename)
    log.setLevel(logging.INFO)
    log_handler = logging.FileHandler(log_file)
    log_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(log_formatter)
    log.addHandler(log_handler)

    return log


def moving_window(numbers: List[float], window_size: int):
    """ Calculates a sliding window of values from the beginning of the list """
    smoothed_data = []

    for i in range(len(numbers)):
        start, end = max(0, i - window_size + 1), i + 1
        window = numbers[start:end]
        smoothed_value = sum(window) / len(window)
        smoothed_data.append(smoothed_value)

    return smoothed_data
