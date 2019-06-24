from data_dealer import DataDealer
from path_set import *
import tensorflow as tf
import os
import numpy
import random

# 屏蔽通知信息和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用VIS(VGG19)+LSTM网络
# VGG19输入的图像数据为224*224*3
class TrainNetForVQA:
    
    def __init__(self):
        # 获取字典
        self.dealer = DataDealer(ANSWERS_DICT_PATH)

    def __loadBatch(self,capacity,batch_size,file_name):
        """
        读取tfrecords的资料
        batch_size为一次取出样本数量，capacity为队列的容量
        """
        def parse(example):
            features = tf.parse_single_example(example,features={      
                                               'answer': tf.FixedLenFeature([self.dealer.getWordNum()], tf.int64),                                                                 
                                               'question' : tf.FixedLenFeature([QUESTION_MAX_LEN * 300], tf.float32),
                                               'img':tf.FixedLenFeature([7*7*512], tf.float32),
                                               'question_len' : tf.FixedLenFeature([1], tf.int64) }) 
            answer = tf.cast(features['answer'],dtype = tf.float32)
            question = tf.cast(features['question'],dtype = tf.float32)
            img = tf.cast(features['img'],dtype = tf.float32)
            question_len = tf.cast(features['question_len'],dtype = tf.int64)
            return answer,question,img,question_len
        for root, dirs, files in os.walk(os.path.dirname(file_name)):  
            pass
        for i in range(len(files)):
            files[i] = os.path.dirname(file_name) + '\\' + files[i]
        dataset = tf.data.TFRecordDataset([files[random.randint(0,len(files)-1)]])
        dataset = dataset.map(parse).repeat().batch(batch_size).shuffle(buffer_size=capacity)
        iterator = dataset.make_one_shot_iterator()
        label_batch,question_batch,img_batch,question_len_batch = iterator.get_next()
        return label_batch,question_batch,img_batch,question_len_batch

    def __get_variables_to_restore(self,str):
      return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if str in v.name]

    def train(self,batch_size,iterate_time,learning_rate):
        """
        训练网络
        """
        model_path = LSTM_MODEL_PATH + LSTM_MODEL_NAME       
        # 生成随机权值
        weights = {
        'w_pic': tf.get_variable('weight_of_pic',initializer=tf.random_normal([512,512])),
        'w_q': tf.get_variable('weight_of_question',initializer=tf.random_normal([300,300])),
        'w_q_out': tf.get_variable('weight_of_question_output',initializer=tf.random_normal([300,512])),
        'w_pos': tf.get_variable('pos_weight_of_question_output',initializer=tf.random_normal([300,49])),
        'w_pic_in': tf.get_variable('weight_of_pic_in_lstm',initializer=tf.random_normal([512,300])),
        'w_q_in': tf.get_variable('weight_of_question_in_lstm',initializer=tf.random_normal([300,300])),
        'out': tf.get_variable('wo',initializer=tf.random_normal([300,self.dealer.getWordNum()]))
        }
        biases = {
            'b_pic': tf.get_variable('biase_of_pic',initializer=tf.random_normal([512,])),
            'b_q': tf.get_variable('biase_of_question',initializer=tf.random_normal([300,])),
            'b_q_out': tf.get_variable('biase_of_question_output',initializer=tf.random_normal([512,])),
            'b_pos': tf.get_variable('pos_biase_of_question_output',initializer=tf.random_normal([49,])),
            'b_pic_in': tf.get_variable('biase_of_pic_in_lstm',initializer=tf.random_normal([300,])),
            'b_q_in': tf.get_variable('biase_of_question_in_lstm',initializer=tf.random_normal([300,])),
            'out': tf.get_variable('bo',initializer=tf.random_normal([self.dealer.getWordNum(),]))
        }

        # 导入数据
        label_batch,question_batch,img_batch,question_len_batch = self.__loadBatch(iterate_time,batch_size,TRAIN_BATCH_PATH)
        # 获取图像数据矩阵
        img_batch = tf.reshape(img_batch,[batch_size*7*7,512])
        img_batch = tf.add(tf.matmul(img_batch, weights['w_pic']), biases['b_pic'])
        img_batch = tf.nn.leaky_relu(img_batch)
        # 空间序矩阵batch_size,49,512
        img_batch = tf.reshape(img_batch,[batch_size,7*7,512])

        # 问题长度
        question_len_batch = tf.reshape(question_len_batch,[batch_size])
        label_batch = tf.reshape(label_batch,[batch_size,self.dealer.getWordNum()])
        question_batch = tf.reshape(question_batch,[batch_size*QUESTION_MAX_LEN,300])
        data = tf.add(tf.matmul(question_batch, weights['w_q']), biases['b_q'])
        data_batch = tf.reshape(data,[batch_size,-1,300])
        # 问题矩阵序列构建
        data_batch = tf.nn.leaky_relu(data_batch)

        # 进入LSTM网络训练

        # LSTM网络得到问题特征
        question_lstm_cell = tf.nn.rnn_cell.LSTMCell(300,name = 'QuestionLSTMCell')
        # 设置dropout
        question_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(question_lstm_cell, input_keep_prob=0.7, output_keep_prob=0.7)
        # 初始状态为0
        q_init_state = question_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        question_outputs, question_states = tf.nn.dynamic_rnn(question_lstm_cell, data_batch, initial_state=q_init_state, dtype = tf.float32,sequence_length = question_len_batch)
        question_output = question_states.h
        # 从问题序列中提取出来的特征
        question_output = tf.reshape(question_output,(batch_size,300))

        # 使用问题提取的特征生成空间权重 [batch_size,49]
        pos_weight = tf.add(tf.matmul(question_output, weights['w_pos']), biases['b_pos'])
        pos_weight = tf.reshape(pos_weight,[1,batch_size*49])
        # sigmoid规范化，这个权值在预想中是用于遗忘空间不需要的特征的
        pos_weight = tf.sigmoid(pos_weight)
        img_batch = tf.transpose(img_batch,(2,0,1))
        img_batch = tf.reshape(img_batch,(512,batch_size*7*7))
        img_batch = img_batch * pos_weight
        img_batch = tf.reshape(img_batch,(512,batch_size,7*7))
        img_batch = tf.transpose(img_batch,(1,2,0))

        # 使用问题提取的特征生成问题权重 [batch_size,512]
        question_weight = tf.add(tf.matmul(question_output, weights['w_q_out']), biases['b_q_out'])
        question_weight = tf.reshape(question_weight,[1,batch_size*512])
        # sigmoid规范化，这个权值在预想中是用于遗忘物体不需要的特征的
        question_weight = tf.sigmoid(question_weight)
        # 遗忘部分物体特征
        img_batch = tf.transpose(img_batch,(1,0,2))
        img_batch = tf.reshape(img_batch,(7*7,batch_size*512))
        img_batch = img_batch * question_weight
        img_batch = tf.reshape(img_batch,[7*7,batch_size,512])
        img_batch = tf.transpose(img_batch,(1,0,2))

        img_lstm_cell = tf.nn.rnn_cell.LSTMCell(512,name = 'PicLSTMCell')
        # 设置dropout
        img_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(img_lstm_cell, input_keep_prob=0.7, output_keep_prob=0.7)
        # 初始状态为0
        img_init_state = img_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        img_outputs, img_states = tf.nn.dynamic_rnn(img_lstm_cell, img_batch, initial_state=img_init_state, dtype = tf.float32)
        img_output = img_states.h

        img_in = tf.add(tf.matmul(img_output, weights['w_pic_in']), biases['b_pic_in'])
        img_in = tf.reshape(img_in,[batch_size,1,300])
        q_in = tf.add(tf.matmul(question_batch, weights['w_q_in']), biases['b_q_in'])
        q_in = tf.reshape(q_in,[batch_size,QUESTION_MAX_LEN,300])
        data_add = tf.concat(axis = 1,values = [img_in,q_in])
        # 约束大小
        data_add = tf.sigmoid(data_add)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(300,name = 'LSTMCell')
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        question_len_batch = question_len_batch + 1
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, data_add, initial_state=init_state, dtype = tf.float32,sequence_length = question_len_batch)

        pred = tf.add(tf.matmul(states.h, weights['out']), biases['out'])
        pred = tf.sigmoid(pred)

        # 防止梯度爆炸
        pred = tf.clip_by_value(pred,1e-7,1.0-1e-7)

        # 计算交叉熵
        # 由于正例远小于负例，使用激励系数
        up = 5
        loss = -(tf.log(pred)*label_batch*up + (1 - label_batch)*tf.log(1 - pred))
        loss = tf.reduce_mean(loss,name = 'loss')

        # 计算准确率
        accuracy = tf.abs(pred - label_batch)
        # print(accuracy)
        accuracy = 1 - tf.reduce_mean(accuracy,name = 'accuracy')

        # 建立优化器 随机梯度下降
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

        # 减少误差，提升准确度
        train = optimizer.minimize(loss)

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # 输入变量
            init = tf.group(tf.global_variables_initializer())
            sess.run(init)
            if os.path.exists(LSTM_MODEL_PATH):
                # 变量替换
                saver.restore(sess, model_path)            
            for m in range(iterate_time):
                sess.run(train)
                ac,lo = sess.run([accuracy,loss])    
                if not iterate_time % (m+1) == 0:
                    continue
                print('loss:',end = '')
                print(lo)
                print('accuracy:',end = '')
                print(ac)
            if not os.path.exists(LSTM_MODEL_PATH):
                os.makedirs(LSTM_MODEL_PATH)
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)



if __name__ == '__main__': 
    net = TrainNetForVQA()
    learning_rate = 1
    learning_rate_down = 0.95
    for i in range(100):
        net.train(batch_size = 10,iterate_time = 1000,learning_rate = learning_rate)
        # 重新设置图
        tf.reset_default_graph()
        # 学习率递减
        learning_rate = learning_rate * learning_rate_down