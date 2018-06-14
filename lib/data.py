#date    :2018-6-4
#brief   :
'''


'''
import pandas as pd  
import numpy as np  
import tensorflow as tf  
from sklearn.metrics import mean_absolute_error,mean_squared_error  
from sklearn.preprocessing import MinMaxScaler  
import matplotlib.pyplot as plt  
import warnings 
import csv  
import sys
import csv
import numpy as np

def read(path,one_hot=True):
    with open(path,'r') as f:
        reader = csv.reader(f);
        next(reader);
        next(reader);
        data=[]
        for row in reader:
            tmp=[]
            if row[1]=='':
                tmp.append(0)
                tmp.append(0)
            else:
                tmp.append(float(row[1]))
                tmp.append(float(row[2]))
            tmp.append(float(row[3]))
            data.append(tmp)
        #print (np.array(data))
        return np.array(data)


def get_data(batch_size=60,time_step=20,train_begin=0,train_end=487):  
    batch_index=[]   
    scaled_x_data=data[:,:-1]
    scaled_y_data=data[:,-1]
    label_train = scaled_y_data[train_begin:train_end]  
    label_test = scaled_y_data[train_end:]  
    normalized_train_data = scaled_x_data[train_begin:train_end]  
    normalized_test_data = scaled_x_data[train_end:]  
      
    train_x,train_y=[],[] 
    for i in range(len(normalized_train_data)-time_step):  
        if i % batch_size==0:  
            batch_index.append(i)  
        x=normalized_train_data[i:i+time_step,:2]  
        y=label_train[i:i+time_step,np.newaxis]  
        train_x.append(x.tolist())  
        train_y.append(y.tolist())  
    batch_index.append((len(normalized_train_data)-time_step)) 
    size=(len(normalized_test_data)+time_step-1)//time_step    
    test_x,test_y=[],[]    
    for i in range(size-1):  
        x=normalized_test_data[i*time_step:(i+1)*time_step,:2]  
        y=label_test[i*time_step:(i+1)*time_step]  
        test_x.append(x.tolist())  
        test_y.extend(y)  
    test_x.append((normalized_test_data[(i+1)*time_step:,:2]).tolist())  
    test_y.extend((label_test[(i+1)*time_step:]).tolist())      

    return batch_index,train_x,train_y,test_x,test_y

path = '../data/1.csv'  
data=read(path)