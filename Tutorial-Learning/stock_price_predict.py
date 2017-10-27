"""
This is a tutoral, and it is downloaded from https://github.com/LouisScorpio/datamining/tree/master/tensorflow-program/rnn/stock_predict
I made some changes in order to predict on the history predictions, not the test_dataset. While the result is not very good. Further modifies
will be make.

This code just for learning the RNN, I declare no copyright of this code, and the copyright belongs to the github user: LouisScorpio.

If I violate your copyright, please contact me at liu.sy.chn@hotmail.com And I will delete this file in time.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


f=open('./dataset/dataset_1.csv')  
df=pd.read_csv(f)     
data=np.array(df['price'])   
data=data[::-1]      
base_path = "/home/dongxq/liu_learning/stock_predict" 

plt.figure()
plt.plot(data)
plt.show()
normalize_data=(data-np.mean(data))/np.std(data)  
normalize_data=normalize_data[:,np.newaxis]       


time_step=20      
rnn_unit=10       
batch_size=50     
input_size=1      
output_size=1     
lr=0.0006

#Making Training Dataset         
train_x,train_y=[],[]
length = len(normalize_data)-50*time_step-1   
for i in range(length):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
#print x,y
print np.shape(train_x)
#The remaining data is using to make comparision 
test_start=[]
a=normalize_data[length+1:length+1+time_step]
test_start.append(a.tolist())
print test_start


X=tf.placeholder(tf.float32, [None,time_step,input_size])   
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }

def lstm(batch):      
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit,reuse=tf.get_variable_scope().reuse)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
    	pred,_=lstm(batch_size)
    
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(10):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                
                if step%10==0:
                    print(i,step,loss_)
                    print("save the model:",saver.save(sess,'stock.model'))
                step+=1
train_lstm()

def prediction():

    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(1)

    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        
        module_file = tf.train.latest_checkpoint(base_path)
        saver.restore(sess, module_file) 

        prev_seq=train_x[-3000]
        predict=[]
        
        for i in range(1000):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            prev_seq.append(next_seq[-1].tolist())
            prev_seq=prev_seq[:]
            prev_seq=prev_seq[1:21]
        
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(length, length + len(predict))), predict, color='r')
        plt.show()

prediction() 
