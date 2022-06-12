import torch
import numpy as np
import random
import csv
import os
import cv2

def read_file(data_path, csv_path):
    training_datas, labels = [], []
    with open(csv_path, newline='') as f:
        rows = csv.reader(f)
        for i, row in enumerate(rows):
            if i != 0:
                name, label = row[0], row[1]
                pic_path = os.path.join(data_path, f'{name}.png')
                pic = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
                training_datas.append(pic)
                labels.append(float(label))
                #print(pic.shape)
                #cv2.imwrite('./132.png', pic)
    return training_datas, labels


def fix_seed():
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

if __name__ == '__main__':
    read_file('./correlation_assignment/images', './correlation_assignment/responses.csv')