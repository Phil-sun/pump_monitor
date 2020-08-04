"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import os

def batch_generator(data,label,batchsize):
    size = data.shape[0]
    data_copy = data.copy()
    label_copy = label.copy()

    indices = np.arange(size)
    np.random.shuffle(indices)
    data_copy = data_copy[indices]
    label_copy = label[indices]

    idx = 0
    while True:
        if (idx+batchsize) <= size:
            yield data_copy[idx:(idx+batchsize)],label_copy[idx:(idx+batchsize)]
            idx += batchsize
        else:
            idx = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            data_copy = data_copy[indices]
            label_copy = label[indices]
            continue


def train():
    # data
    # mnist = input_data.read_data_sets('./mnist', one_hot=True)              # they has been normalized to range (0,1)
    # test_x = mnist.test.images[:2000]
    # test_y = mnist.test.labels[:2000]
    # print("test_x.shape:",test_x.shape)

    '''一、读取文件'''
    # csv文件路径，放在当前py文件统一路径(存放路径自己选择)
    train_file_path = "data/mulclass.csv"
    all_data = pd.read_csv(train_file_path,encoding='gbk')
    all_data.fillna(value=0, inplace=True)
    # item_data = all_data.drop(['Date',' Time','predict'], axis=1)
    # item_data = all_data.drop(['Date', ' Time', 'predict', '泵入口温度', '电机温度', '油温'], axis=1)
    # item_data = all_data.drop(['Date', ' Time', '泵入口温度', '电机温度', '油温'], axis=1)
    item_data = all_data
    feat_labels = item_data.columns
    print("item_data.shape:",item_data.shape)
    print("len(feat_labels):",len(feat_labels))

    df_0 = item_data[item_data['predict']==100]
    df_1 = item_data[item_data['predict']!=100]

    # 合成100的数据块，第一维缩小为原来1/100，合成6次，和正样本对齐
    data_choice = []
    for i in range(6):
        d = df_0.values[i:(df_0.shape[0]//100*100)+i]
        # df_1 = df_1.iloc[:(df_1.shape[0]//100*100)]
        dd = d.reshape([-1,100,len(feat_labels)])
        print("d.shape[0]:", d.shape[0])
        print("dd.shape:",dd.shape)
        data_choice.append(dd)
    # index_0 = np.random.choice(data_0.shape[0],size=int(df_1.shape[0]),replace=False)
    # data_choice = data_0[index_0].reshape([-1,len(feat_labels)])
    print("data_choice.shape:", np.shape(data_choice))
    data_choice = np.array(data_choice).reshape([-1,len(feat_labels)])

    # 负样本分割数据和标签
    data_0_x = data_choice[:, :-1].reshape([-1,100,len(feat_labels)-1])
    data_0_y = data_choice[:,-1].reshape([-1,100]).max(axis=1)

    # 正样本分割数据和标签
    data_1_x = []
    data_1_y = df_1['predict'].values
    df_1.drop(['predict'],axis=1,inplace=True)
    for i in range(100,df_1.shape[0]+1):
        # if data_1_x.size == 0:
        #     data_1_x = df_1[i-100:i].values
        # else:
        #     data_1_x = np.append(data_1_x,df_1[i-100:i].values,axis=0)
        data_1_x.append(df_1[i-100:i].values)

    # print("data_1_x[0]:",data_1_x[0])
    # print("data_1_x.shape:", np.shape(data_1_x))
    data_1_x = np.array(data_1_x)
    print("data_1_x[0]:",data_1_x[0])
    print("data_1_x.shape:",data_1_x.shape)


    data_1_y = data_1_y[99:]

    data_x = np.concatenate([data_0_x,data_1_x],axis=0)
    data_y = np.concatenate([data_0_y,data_1_y],axis=0)

    # df_process = pd.DataFrame(data_choice,columns=feat_labels)
    # df_balance = pd.concat([df_process,df_1],ignore_index=True)
    # b_y = df_balance['predict']
    # print("b_y:",b_y.value_counts())
    # print("b_y:",b_y.value_counts(normalize=True))
    # # df_balance.to_csv("data/6-CFD22-1-A37H_balance.csv",index=False)
    #
    # # col_t = item_data.shape[0] // 100 * 100
    # # col_t = 1000
    # # block_data = np.reshape(item_data.values[:col_t,:],[-1,100,5,1])
    # df_feature = df_balance.drop(['predict'],axis=1)
    # block_data = np.reshape(df_feature.values, [-1, 100, 5])
    # input_y = np.reshape(df_balance['predict'].values,[-1,100]).max(axis=1)
    # print("input_y:",pd.Series(input_y).value_counts())
    # print("input_y:", pd.Series(input_y).value_counts(normalize=True))
    data_y = data_y[:,np.newaxis]
    print("data_x.shape:",data_x.shape)
    print("data_y.shape:",data_y.shape)
    train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.3)
    print("test_y.shape:",test_y.shape)
    print("test_y:",pd.Series(np.squeeze(test_y,axis=1)).value_counts())
    print("test_y:", pd.Series(np.squeeze(test_y,axis=1)).value_counts(normalize=True))


    # encoder = LabelBinarizer()
    onehot = OneHotEncoder(sparse=False)
    # print("train_y:",train_y[:10])
    # print("test_y:",test_y[:10])
    train_y = onehot.fit_transform(train_y)
    test_y = onehot.fit_transform(test_y)
    print("train_y.shape:", train_y.shape)
    print("train_y:",train_y[:10])
    print("test_y:",test_y[:10])
    class_len = train_y.shape[1]




    tf.set_random_seed(1)
    np.random.seed(1)

    # Hyper Parameters
    BATCH_SIZE = 1000
    TIME_STEP = 100          # rnn time step / image height
    INPUT_SIZE = train_x.shape[2]         # rnn input size / image width
    LR = 0.01               # learning rate


    # plot one example
    # print(mnist.train.images.shape)     # (55000, 28 * 28)
    # print(mnist.train.labels.shape)   # (55000, 10)
    # plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
    # plt.title('%i' % np.argmax(mnist.train.labels[0]))
    # plt.show()

    # tensorflow placeholders
    # tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # shape(batch, 784)
    # image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)
    tf_x = tf.placeholder(tf.float32, [None, TIME_STEP,INPUT_SIZE])       # shape(batch, 784)
    image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])
    tf_y = tf.placeholder(tf.int32, [None, class_len])                             # input y

    # RNN
    # rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
    # outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    #     rnn_cell,                   # cell you have chosen
    #     image,                      # input
    #     initial_state=None,         # the initial hidden state
    #     dtype=tf.float32,           # must given if set initial_state = None
    #     time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
    # )initial_state
    num_units = [64,128,256]
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]
    mul_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    # state = mul_cells.zero_state(BATCH_SIZE,tf.float32)
    outputs, states = tf.nn.dynamic_rnn(
        mul_cells,                   # cell you have chosen
        image,                      # input
        initial_state=None,         # the initial hidden state
        dtype=tf.float32,           # must given if set initial_state = None
        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
    )

    output = tf.layers.dense(outputs[:, -1, :], class_len)              # output based on the last output step

    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
        labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph

    saver = tf.train.Saver(max_to_keep=5)
    gen = batch_generator(train_x,train_y,BATCH_SIZE)
    for step in range(2000):    # training
        # b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        b_x, b_y = next(gen)
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        if step % 50 == 0:      # testing
            # print('#' * 100)
            # print("b_x.shape:", b_x.shape)
            accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
            print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
        if step % 500 == 0:
            saver.save(sess,"model/lstm_model/save_net.ckpt",global_step=step)
            print("save model successfully")

    # print 10 predictions from test data
    test_output = sess.run(output, {tf_x: test_x[:10]})
    pred_y = np.argmax(test_output, axis=1)
    print(pred_y, 'prediction number')
    print(np.argmax(test_y[:10], axis=1), 'real number')

    print("model finish")



def data_prepare():
    file_list = ["org_data/"+f for f in os.listdir("org_data") if os.path.isfile("org_data/"+f)]
    col=['油压','泵出口压力','泵入口温度','A相运行电流','泵入口压力','电机温度',	'运行频率','漏电电流','运行电压','X轴震动','油温','油嘴开度']
    df_all = pd.DataFrame()
    value_list = [10, 40, 50, 60, 70, 80, 90,100]
    for f in file_list:
        print("f:",f)
        df_f = pd.read_csv(f,header=[0,1],encoding="gbk")
        print("df_f.shape:",df_f.shape)
        df_f.columns = df_f.columns.droplevel(1)
        print("df_f.columns:", df_f.columns)
        df_f = df_f[col]
        df_f["predict"] = 100
        for i in range(len(value_list)-1):
            if i == 0:
                df_f.iloc[-1440:, -1] = value_list[i]
            else:
                df_f.iloc[-1440*(i+1):-1440*i,-1] = value_list[i]

        df_all = pd.concat([df_all,df_f],ignore_index=True,axis=0) if not df_all.empty else df_f

    # 处理异常值
    print("df_all.shape:", df_all.shape)
    df_all.dropna(inplace=True)
    print("df_all.shape:",df_all.shape)
    df_1 = df_all[df_all["predict"]==100]
    print("df_1.shape:",df_1.shape)
    df_2 = df_all[df_all["predict"]!=100]
    print("df_2.shape:", df_2.shape)
    df_1 = df_1[~(df_1==0).any(axis=1)]
    print("df_1.shape:", df_1.shape)

    df_pro = pd.concat([df_1,df_2],ignore_index=True)
    print("df_pro.shape:",df_pro.shape)
    df_pro.to_csv("data/mulclass.csv",index=False,encoding='gbk')

if __name__ == '__main__':
    # data_prepare()
    train()
