import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
signal_size = 1024
datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
                   "Normal Baseline Data"]
axis = ["_DE_time", "_FE_time", "_BA_time"]
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm
label = [i for i in range(0, 10)]

def dataset_information(source_N, target_N, label_inconsistent):
    if label_inconsistent == 'PADA':
        label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if (source_N == [0] and target_N == [1]) or (source_N == [0] and target_N == [2]):
                label_target = [0, 1, 2, 4, 5, 7, 8, 9]
        name_source = label_source
        name_target = label_target
    if label_inconsistent == 'OSBP':
        name_target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if (source_N == [0] and target_N == [1]) or (source_N == [0] and target_N == [2]):
                name_source =  [0, 2, 3, 5, 6, 7, 8, 9]
                label_source = [0, 1, 2, 3, 4, 5, 6, 7]
                label_target = [0, 8, 1, 2, 8, 3, 4, 5, 6, 7]
    if label_inconsistent == 'UAN':
        if (source_N == [0] and target_N == [1]) or (source_N == [0] and target_N == [2]):
                name_source =  [0, 1, 2, 4, 5, 6, 7, 8, 9]
                label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                name_target =  [1, 2, 3, 4, 5, 7, 8, 9]
                label_target = [1, 2, 9, 3, 4, 6, 7, 8]
    num_classes = len(label_source)
    return name_source, name_target, label_source, label_target, num_classes

def get_files(root, N, name, label):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for i, n in enumerate(name):
            #print(n)
            if int(dataname[N[k]][n].split(".")[0])<101:
                path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[i])
            data += data1
            lab +=lab1

    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class CWRU_inconsistent(object):
    #num_classes = 10
    inputchannel = 1
    def __init__(self, data_dir, transfer_task,inconsistent, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.inconsistent = inconsistent
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
            name_source, name_target, label_source, label_target, num_classes = dataset_information(self.source_N,self.target_N,self.inconsistent)
            list_data = get_files(self.data_dir, self.source_N, name_source, label_source)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N, name_target, label_target)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val, num_classes
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


"""
    def data_split(self):

"""