
#author: honghuichao
#date  :2018-6-
#brief :
import pandas as pd  
import numpy as np  
import tensorflow as tf  
from sklearn.metrics import mean_absolute_error,mean_squared_error  
from sklearn.preprocessing import MinMaxScaler  
import matplotlib.pyplot as plt  
import warnings 
import csv  
import sys
sys.path.append('../')
from lib.data import read,get_data
from config import *
warnings.filterwarnings('ignore')

'''

define network and initial it's weights  and bias

'''
tf.reset_default_graph()    
weights={  
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),  
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))  
         }  
biases={  
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),  
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))  
        }  

def lstm(X):    
    '''
    main construct ：
    '''
    batch_size=tf.shape(X)[0]  
    time_step=tf.shape(X)[1]  
    w_in=weights['in']  
    b_in=biases['in'] 
    input=tf.reshape(X,[-1,input_size])    
    input_rnn=tf.matmul(input,w_in)+b_in  
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])    
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)  
    #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)  
    init_state=cell.zero_state(batch_size,dtype=tf.float32)  
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果  
    output=tf.reshape(output_rnn,[-1,rnn_unit])  
    w_out=weights['out']  
    b_out=biases['out']  
    pred=tf.matmul(output,w_out)+b_out  
    return pred,final_states 

def train_lstm(batch_size=80,time_step=10,train_begin=0,train_end=200):  
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])  
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])  
    batch_index,train_x,train_y,test_x,test_y = get_data(batch_size,time_step,train_begin,train_end)  
    pred,_=lstm(X)  
    loss=tf.reduce_mean(tf.nn.cross_entropy(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))  
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)    
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())   
        iter_time = 5000 
        for i in range(iter_time):  
            for step in range(len(batch_index)-1):  
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})  
            if i % 100 == 0:      
                print('iter:',i,'loss:',loss_)  

        test_predict=[] 

        '''
        #set a random array from test_set as test sample 
        a=[]
        a.append([427553526,873823325])
        a.append([368481908,765448979])
        a.append([340887778,658621406])
        a.append([436080245,876021500])
        a.append([427563540,855395668])
        a.append([437238983,864842479])
        a.append([425245740,811475164])
        a.append([423994480,806593633])
        a.append([456750184,820638739])
        a.append([465628921,871084241])
        a.append([487953591,904534272])
        a.append([403112605,835578169])
        a.append([454568260,927418827])
        a.append([447724781,931109894])
        a.append([436586388,950130396])
        '''
        prob=sess.run(pred,feed_dict={X:[a]}) 
        predict=prob.reshape((-1))  
        print(predict)
        
        for step in range(len(test_x)):  
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})     
            predict=prob.reshape((-1))  
            test_predict.extend(predict)  

        rmse=np.sqrt(mean_squared_error(test_predict,test_y))  
        mae = mean_absolute_error(y_pred=test_predict,y_true=test_y)  
        print ('mae:',mae,'   rmse:',rmse) 
        
    return test_predict  

'''
main processed
train and test our network

'''

test_predict = train_lstm(batch_size=800,time_step=150,train_begin=0,train_end=487)  
