import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit=10
input_size=4
output_size=1
lr=0.0005
layer_num=2

f=open('YourPath')
df=pd.read_csv(f)
data=df.iloc[:,:4].values
base_path="YourPath"

normalized_data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
mean=np.mean(data,axis=0)[0]
std=np.std(data,axis=0)[0]
mean_=mean.tolist()
std_=std.tolist()
print mean_
print std_
average=[1,2,3,4,5,6,7,8,9,10,11,12]
my_loss=[]

def get_train_data(batch_size=10,time_step=12,train_begin=0,train_end=500):
    batch_index=[]
    data_train=normalized_data[train_begin:train_end]
    train_x,train_y=[],[]
    #batch_index.append(0)
    for i in range(len(data_train)-2*time_step):
        if i%batch_size==0:
            batch_index.append(i)
        x=data_train[i:i+time_step,:4]
        y=data_train[i+time_step:i+time_step+1,0,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    return batch_index,train_x,train_y

def get_test_data(time_step=12,test_begin=500,test_end=516):
    count=(test_begin+1)%12
    if count==0:
        count=12
    test_data=normalized_data[test_begin:test_end]
    test_x,test_y=[],[]
    test_x=test_data[0:12,:4]
    test_y=test_data[12:13,0:1]
    return test_x,test_y,count

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,rnn_unit]))
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
    final_out=tf.reshape(pred,[-1,time_step,rnn_unit])
    final_out=final_out[:,-1,-1]
    return final_out,final_states

def train_lstm(batch_size=10,time_step=12,train_begin=0,train_end=500):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,1,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm"):
        pred,final=lstm(X)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                print(i,loss_)
                
            if i % 200==0:
                print("save_model",saver.save(sess,'rainfall.model',global_step=i))
            if i % 100==0:
                my_loss.append(loss_)
                
train_lstm()

def pearson(pre,real):
    x=np.empty(shape=[2,4])
    for i in range(4):
        x[0][i]=pre[i]
    for j in range(4):
        x[1][j]=real[j]
    y=np.corrcoef(x)
    return y[0][1]

def prediction(time_step=12):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    test_x,test_y,count=get_test_data(time_step)   
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,final=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(base_path)
        saver.restore(sess, module_file) 
        test_predict=[]
        prob=sess.run(pred,feed_dict={X:[test_x]})   
        predict=prob.reshape((-1))
        test_predict.extend(predict)
        real_value=test_y[0]
        for i in range(1):
            test_predict[i]=test_predict[i]*std_+mean_
            real_value[i]=test_y[i]*std_+mean_
        count=count-1
        print("The Predictions is:")
        print test_predict[0]
        print("The real value is:")
        print real_value[0]
        print("The average is:")
        print average[count]
        plt.figure()
        plt.plot(list(range(len(my_loss))), my_loss,label='Average',color='y')
        plt.show()
       
prediction()