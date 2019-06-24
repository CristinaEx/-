from yuki_visual import YukiVisual
from data_reader import DataReader
from data_dealer import DataDealer
from tensorflow.python.ops import rnn_cell_impl
from PIL import Image
from path_set import *
from vgg19 import VGG19model
import tensorflow as tf
import os
import numpy
import gensim

class Tester(YukiVisual):

    def __init__(self):
        # 获取字典
        self.dealer = DataDealer(ANSWERS_DICT_PATH)
        if not os.path.exists(LSTM_MODEL_PATH):
            exit()
        self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_WIKI_GENSIM_DATA_PATH)
        self.img = None
        self.path = '.\\'
        YukiVisual.__init__(self)       

    def __cd(self,path_):
        """
        成功进入目标文件夹返回True，否则返回False
        """
        # 创建目录
        if not os.path.exists(self.path + path_):
            return False
        else:
            self.path = self.path + path_ + '\\'
        return True

    def __op(self,image_name):
        """
        成功打开图像返回True，否则返回False
        """
        if not os.path.exists(self.path + image_name ):
            return False
        else:
            self.img = Image.open(self.path+image_name)
            self.img.show()
            self.img = self.img.resize((224,224))
            self.img = numpy.array(self.img)
        return True

    def __getAnswer(self,question):
        """
        获取question对应的答案
        """
        try:
            self.img.shape
        except:
            return "未打开有效图片!"

        # 重新设置图
        tf.reset_default_graph()
        
        img_batch = tf.Variable(self.img,dtype = tf.float32)
        img_batch = tf.reshape(img_batch,[1,224,224,3])
        # 获取VGG19网络得到的结果
        img_batch = self.__getVGG19Result(img_batch)

        # 重新设置图
        tf.reset_default_graph()

        img_batch = tf.cast(img_batch,tf.float32)

        question = question.replace('\n','')
        question = question.replace('?','')
        question = question.replace(',',' ,')
        question = question.replace('.',' .')
        question = question.split(' ')
        data = []
        for word in question:
            # data.shape = (len(question),300)
            try:
                data.append(list(self.model_word2vec[word]))
            except:
                data.append([0]*300)
        question_batch = tf.cast(data,tf.float32)

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

        # 获取图像数据矩阵
        img_batch = tf.reshape(img_batch,[7*7,512])
        img_batch = tf.add(tf.matmul(img_batch, weights['w_pic']), biases['b_pic'])
        img_batch = tf.nn.leaky_relu(img_batch)
        # 空间序矩阵1,49,512
        img_batch = tf.reshape(img_batch,[1,7*7,512])

        question_batch = tf.reshape(question_batch,[1*len(question),300])
        data = tf.add(tf.matmul(question_batch, weights['w_q']), biases['b_q'])
        data_batch = tf.reshape(data,[1,-1,300])
        # 问题矩阵序列构建
        data_batch = tf.nn.leaky_relu(data_batch)

        # 进入LSTM网络训练

        # LSTM网络得到问题特征
        question_lstm_cell = tf.nn.rnn_cell.LSTMCell(300,name = 'QuestionLSTMCell')
        # 设置dropout
        question_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(question_lstm_cell, input_keep_prob=0.7, output_keep_prob=0.7)
        # 初始状态为0
        q_init_state = question_lstm_cell.zero_state(1, dtype=tf.float32)
        question_outputs, question_states = tf.nn.dynamic_rnn(question_lstm_cell, data_batch, initial_state=q_init_state, dtype = tf.float32)
        question_output = question_states.h
        # 从问题序列中提取出来的特征
        question_output = tf.reshape(question_output,(1,300))

        # 使用问题提取的特征生成空间权重 [1,49]
        pos_weight = tf.add(tf.matmul(question_output, weights['w_pos']), biases['b_pos'])
        pos_weight = tf.reshape(pos_weight,[1,1*49])
        # sigmoid规范化，这个权值在预想中是用于遗忘空间不需要的特征的
        pos_weight = tf.sigmoid(pos_weight)
        img_batch = tf.transpose(img_batch,(2,0,1))
        img_batch = tf.reshape(img_batch,(512,1*7*7))
        img_batch = img_batch * pos_weight
        img_batch = tf.reshape(img_batch,(512,1,7*7))
        img_batch = tf.transpose(img_batch,(1,2,0))

        # 使用问题提取的特征生成问题权重 [1,512]
        question_weight = tf.add(tf.matmul(question_output, weights['w_q_out']), biases['b_q_out'])
        question_weight = tf.reshape(question_weight,[1,1*512])
        # sigmoid规范化，这个权值在预想中是用于遗忘物体不需要的特征的
        question_weight = tf.sigmoid(question_weight)
        # 遗忘部分物体特征
        img_batch = tf.transpose(img_batch,(1,0,2))
        img_batch = tf.reshape(img_batch,(7*7,1*512))
        img_batch = img_batch * question_weight
        img_batch = tf.reshape(img_batch,[7*7,1,512])
        img_batch = tf.transpose(img_batch,(1,0,2))

        img_lstm_cell = tf.nn.rnn_cell.LSTMCell(512,name = 'PicLSTMCell')
        # 设置dropout
        img_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(img_lstm_cell, input_keep_prob=0.7, output_keep_prob=0.7)
        # 初始状态为0
        img_init_state = img_lstm_cell.zero_state(1, dtype=tf.float32)
        img_outputs, img_states = tf.nn.dynamic_rnn(img_lstm_cell, img_batch, initial_state=img_init_state, dtype = tf.float32)
        img_output = img_states.h

        img_in = tf.add(tf.matmul(img_output, weights['w_pic_in']), biases['b_pic_in'])
        img_in = tf.reshape(img_in,[1,1,300])
        q_in = tf.add(tf.matmul(question_batch, weights['w_q_in']), biases['b_q_in'])
        q_in = tf.reshape(q_in,[1,len(question),300])
        data_add = tf.concat(axis = 1,values = [img_in,q_in])
        # 约束大小
        data_add = tf.sigmoid(data_add)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(300,name = 'LSTMCell')
        init_state = lstm_cell.zero_state(1, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, data_add, initial_state=init_state, dtype = tf.float32)

        pred = tf.add(tf.matmul(states.h, weights['out']), biases['out'])
        pred = tf.sigmoid(pred)

        # 获取saver
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # 输入变量
            init = tf.group(tf.global_variables_initializer())
            sess.run(init)
            # 变量替换
            saver.restore(sess, model_path)
            re = sess.run(pred)
        return list(re)[0]

    def __getVGG19Result(self,img_batch):
        """
        获取VGG19网络的结果
        img_batch:tf.Variable->[1,224,224,3]
        返回最后一层隐藏层的一维列表
        """
        # 导入VGG19模型
        model = VGG19model()
        weight_vgg,biase_vgg = model.loadWeightsAndBiases(VGG19_WEIGHTS_AND_BIASE_PATH,False)
        # 这里输出的是最后一个隐藏层
        out = model.getNet(img_batch,weight_vgg,biase_vgg,0.2,True)
        init = tf.global_variables_initializer() 
        with tf.Session() as sess:
            sess.run(init)
            data = sess.run(out)
            # 一维列表
            data = data[0]
        return data

    def send_message(self,text):
        text_list = self.received_text.split()
        if text_list[0] == 'cd':
            if self.__cd(text_list[1]):
                YukiVisual.send_message(self,'now in ' + self.path + '\n')
            else:
                YukiVisual.send_message(self,'wrong!\n') 
        elif text_list[0] == 'op':
            if self.__op(text_list[1]):
                YukiVisual.send_message(self,'success!\n Please ask me questions.\n')
            else:
                YukiVisual.send_message(self,'wrong!\n') 
        elif text_list[0] == 'ls':
            a = str()
            for i in os.listdir(self.path):
                a = a + i + '\n '
            YukiVisual.send_message(self,a)
        else:
            data = self.__getAnswer(self.received_text)
            if not type(data) == numpy.ndarray:
                YukiVisual.send_message(self, 'wrong!\n')
                return
            data = list(data)
            sort_list = data[:]
            sort_list.sort()

            # print(sum(sort_list)/len(sort_list))
            sort_list = sort_list[-5:]
            YukiVisual.send_message(self, 'Top Five Answer:\n')
            for i in range(len(data)):
                if data[i] in sort_list:
                    YukiVisual.send_message(self,'*'*int(1+data[i]*10) + ' '*int(11-data[i]*10) + self.dealer.getWord(i) + '      ' +  str(data[i]) + '\n')

if __name__ == '__main__':
    Tester()
    # op coco_qa\images\test2015\COCO_test2015_000000000191.jpg
    # COCO_test2015_000000000219.jpg
    # COCO_test2015_000000006598.jpg
    # COCO_test2015_000000014889.jpg