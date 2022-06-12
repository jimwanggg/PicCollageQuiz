import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import os
import logging

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import read_file, fix_seed
from dataset import ImgDataset
from model import CNN_net

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, help="training and testing data path", default='./correlation_assignment/images')
    parser.add_argument("-l", "--label_path", type=str, help="label path", default='./correlation_assignment/responses.csv')
    parser.add_argument("-m", "--ckpt_path", type=str, help="model save path", default='./checkpoints/')
    parser.add_argument("-e", "--epochs", type=int, help="label path", default=50)
    args = parser.parse_args()
    return args

    
def training(train_loader, test_loader, config):
    logging.basicConfig(filename='summary.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN_net().to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('parameters: {}'.format(pytorch_total_params))
    logging.info('parameters: {}'.format(pytorch_total_params))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
    val_every_step = 1
    train_losses = []
    test_losses = []

    # do eval before training
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, label = data
            img = img.to(device)
            pred = model(img)
            
            loss = loss_func(pred.squeeze(1).float(), label.to(device).float())

            total_loss += loss.item()
    print('Before Training Test Loss: {}'.format(total_loss/len(test_loader)))
    logging.info('Before Training Test Loss: {}'.format(total_loss/len(test_loader)))
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        cur_time = time.time()
        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img, label = data
            img = img.to(device)
            pred = model(img)
            # print(pred.shape, label.shape)
            loss = loss_func(pred.squeeze(1).float(), label.to(device).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print('Epoch [{}/{}], Training Time: {:.3f}, Loss: {}'.format(epoch+1, config.epochs, time.time()-cur_time, total_loss/len(train_loader)))
        logging.info('Epoch [{}/{}], Training Time: {:.3f}, Loss: {}'.format(epoch+1, config.epochs, time.time()-cur_time, total_loss/len(train_loader)))
        train_losses.append(total_loss/len(train_loader))
        if (epoch+1) % val_every_step == 0:
            model.eval()
            total_loss = 0.0
            cur_time = time.time()
            with torch.no_grad():
                for i, data in enumerate(tqdm(test_loader)):
                    img, label = data
                    img = img.to(device)
                    pred = model(img)
                    
                    loss = loss_func(pred.squeeze(1).float(), label.to(device).float())

                    total_loss += loss.item()
            print('Epoch [{}/{}], Testing Time: {:.3f}, Loss: {}'.format(epoch+1, config.epochs, time.time()-cur_time, total_loss/len(test_loader)))
            logging.info('Epoch [{}/{}], Testing Time: {:.3f}, Loss: {}'.format(epoch+1, config.epochs, time.time()-cur_time, total_loss/len(test_loader)))
            torch.save(model.state_dict(), os.path.join(config.ckpt_path, "{}.ckpt".format(epoch+1)))
            test_losses.append(total_loss/len(test_loader))
            plt.plot(list(range(epoch+1)), train_losses, label='train loss', color='red')
            plt.plot(list(range(epoch+1)), test_losses, label='test loss', color='blue')
            plt.xlabel('Epochs')
            plt.ylabel('loss')
            plt.legend()
            
            plt.savefig('./loss.png')
            plt.clf()

    plt.plot(list(range(config.epochs)), train_losses, label='train loss', color='red')
    plt.plot(list(range(config.epochs)), test_losses, label='test loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    
    plt.savefig('./loss.png')
    plt.clf()

if __name__ == '__main__':
    fix_seed()

    config = get_config()


    datas, labels = read_file(config.data_path, config.label_path)

    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.33, random_state=42)

    train_set = ImgDataset(X_train, y_train)
    test_set = ImgDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    training(train_loader, test_loader, config)



