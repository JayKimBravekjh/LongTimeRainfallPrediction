"""
This is a tutoral, and it is downloaded from https://github.com/LouisScorpio/datamining/tree/master/tensorflow-program/rnn/stock_predict
I made some changes in order to predict on the history predictions, not the test_dataset. While the result is not very good. Further modifies
will be made.

This code just for learning the RNN, I declare no copyright of this code, and the copyright belongs to the github user: LouisScorpio.

If I violate your copyright, please contact me at liu.sy.chn@hotmail.com And I will delete this file in time.
"""
# In the project of N-D input, we can not use the same method, cause we can not predict the other factors which are the input data.
# It means we can not like the 1-D input to predict anymore. Step-by-step prediction is impossible.
# A way to solve this problem is to predic 1 time_step. We can change the prediction of the time-length by changing the length of the input.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


rnn_unit=10       
input_size=7
output_size=1
lr=0.0006         
layer_num=2

f=open('./dataset/dataset_2.csv') 
df=pd.read_csv(f)    
data=df.iloc[:,2:10].values  
base_path = "Your Path" 


def get_train_data(batch_size=60,time_step=90,train_dataset_begin=0,train_dataset_end=5931):
    batch_index=[]
    data_train=data[train_dataset_begin:train_dataset_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    print np.shape(normalized_train_data)  
    train_x,train_y=[],[]   
    for i in range(len(normalized_train_data)-2*time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:7]
       y=normalized_train_data[i+time_step:i+2*time_step,7,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y




def get_test_data(time_step=20,test_begin=5931):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  
    size=(len(normalized_test_data)+time_step-1)//time_step  
    test_x,test_y=[],[]
  
    x=normalized_test_data[0:time_step,:7]
    y=normalized_test_data[90:90+time_step,7]
    test_x.append(x.tolist())
    test_y.extend(y)
    #test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    #test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())

    return mean,std,test_x,test_y


weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }


def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit,reuse=tf.get_variable_scope().reuse)
    multi_cell=tf.contrib.rnn.MultiRNNCell([cell for _ in range(layer_num)], state_is_tuple=True)
    init_state=multi_cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(multi_cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states




def train_lstm(batch_size=80,time_step=90,train_begin=0,train_end=5931):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
  
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint(base_path)    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        
        for i in range(2000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 200==0:
                print("save_model",saver.save(sess,'stock2.model',global_step=i))

train_lstm()

def prediction(time_step=90):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    
    mean,std,test_x,test_y=get_test_data(time_step)
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
       
        module_file = tf.train.latest_checkpoint(base_path)
        saver.restore(sess, module_file) 
        test_predict=[]
        prob=sess.run(pred,feed_dict={X:[test_x[0]]})   
        predict=prob.reshape((-1))
        test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        #acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  
        print test_predict
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

prediction() 
