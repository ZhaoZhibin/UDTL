import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
import numpy as np
from tqdm import tqdm

signal_size = 1024


ADBdata_source=[['KA05'],['KA03'],['KI03']]
ADBdata_target=[['KA01','KA07'],['KA08'],['KI01']]
label=[i for i in range(3)]

#3 Bearings with real damages caused by accelerated lifetime tests(14x)

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
#state = WC[0] #WC[0] can be changed to different working states

#generate Training Dataset and Testing Dataset
def get_files(root, N):
    work_condition=0
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    if N[0]==0:
        state = WC[work_condition]  # WC[0] can be changed to different working states
        for j in range(len(ADBdata_source)):
            state1=ADBdata_source[j]
            for k in range(len(state1)):
                for w3 in range(1):
                        name3 =   state + "_"+state1[k]+"_"+ str(w3 + 1)
                        path3 = os.path.join('/tmp', root,state1[k] , name3 + ".mat")
                        data3, lab3= data_load(path3,name=name3,label=label[j])
                        data += data3
                        lab += lab3
    elif N[0]==1:
        state = WC[work_condition]  # WC[0] can be changed to different working states
        for j in range(len(ADBdata_target)):
            state1 = ADBdata_target[j]
            for k in range(len(state1)):
                for w3 in range(1):
                    name3 =   state + "_"+state1[k]+"_"+ str(w3 + 1)
                    path3 = os.path.join('/tmp', root,state1[k] , name3 + ".mat")
                    data3, lab3 = data_load(path3, name=name3, label=label[j])
                    #if w3 == 0 and state1[k] == 'KI01':
                        #print(data3)
                    data += data3
                    lab += lab3

    return [data,lab]

def data_load(filename,name,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  #Take out the data
    fl = fl.reshape(-1,)
    data=[]
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        x = fl[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1,1)
        data.append(x)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class PUFFT_type(object):
    num_classes = 3
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

