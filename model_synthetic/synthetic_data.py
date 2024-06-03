import random
import math
import numpy as np

# create 300 labels of change and no change
class synthetic_data():

    def __init__(self):

        self.change_label = [1]*300
        self.no_change_label = [0]*300

    def getdata(self,training_split:dict):

        random.seed(10)

        all_data = self.change_label + self.no_change_label
        random.shuffle(all_data)
        
        n_train = math.ceil(len(all_data)*training_split['train'])

        train_data = all_data[:n_train]
        test_data = all_data[n_train:]

        return train_data, test_data
    
class np_synthetic_gen():

    def __init__(self,data,training=False):
        self.data = data
        self.training = training

    def __call__(self):

        numparr = []
        for a in self.data:
            if a == 0:
                arr = np.zeros((400,30,125,1))
                numparr.append(arr)
            else:
                arr1 = np.zeros((200,30,125,1))
                arr2 = np.ones((200,30,125,1))
                arr = np.concatenate([arr1,arr2],axis=0)
                numparr.append(arr)
        
        combine = list(zip(numparr,self.data))

        if self.training:
            random.shuffle(combine)
        
        for target, label in combine:
            yield target, label