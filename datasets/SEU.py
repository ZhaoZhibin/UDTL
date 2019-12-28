import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from itertools import islice



#Digital data was collected at 12,000 samples per second
signal_size = 1024
work_condition=['_20_0.csv','_30_2.csv']
dataname= {0:[os.path.join('bearingset','health'+work_condition[0]),
              os.path.join('gearset','Health'+work_condition[0]),
              os.path.join('bearingset','ball'+work_condition[0]),
              os.path.join('bearingset','outer'+work_condition[0]),
              os.path.join('bearingset', 'inner' + work_condition[0]),
              os.path.join('bearingset', 'comb' + work_condition[0]),
              os.path.join('gearset', 'Chipped' + work_condition[0]),
              os.path.join('gearset', 'Miss' + work_condition[0]),
              os.path.join('gearset', 'Surface' + work_condition[0]),
              os.path.join('gearset', 'Root' + work_condition[0]),
              ],
         1:[os.path.join('bearingset','health'+work_condition[1]),
              os.path.join('gearset','Health'+work_condition[1]),
              os.path.join('bearingset','ball'+work_condition[1]),
              os.path.join('bearingset','outer'+work_condition[1]),
              os.path.join('bearingset', 'inner' + work_condition[1]),
              os.path.join('bearingset', 'comb' + work_condition[1]),
              os.path.join('gearset', 'Chipped' + work_condition[1]),
              os.path.join('gearset', 'Miss' + work_condition[1]),
              os.path.join('gearset', 'Surface' + work_condition[1]),
              os.path.join('gearset', 'Root' + work_condition[1]),
              ]
          }

label = [i for i in range(0, 9)]

def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root, dataname[N[k]][n])
            if n==0:
                data1, lab1 = data_load(path1,  label=label[n])
            else:
                data1, lab1 = data_load(path1, label=label[n-1])
            data += data1
            lab +=lab1

    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    #--------------------
    f = open(filename, "r", encoding='gb18030', errors='ignore')
    fl = []
    if  "ball_20_0.csv" in filename:
        for line in islice(f, 16, None):  # Skip the first 16 lines
            line = line.rstrip()
            word = line.split(",", 8)  # Separated by commas
            fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
    else:
        for line in islice(f, 16, None):  # Skip the first 16 lines
            line = line.rstrip()
            word = line.split("\t", 8)  # Separated by \t
            fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
    #--------------------
    fl = np.array(fl)
    fl = fl.reshape(-1, 1)
    # print(fl.shape())
    data = []
    lab = []
    start, end = int(fl.shape[0]/2), int(fl.shape[0]/2)+signal_size
    while end <= (int(fl.shape[0]/2)+int(fl.shape[0]/3)):
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class Md(object):
    num_classes = 9
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val