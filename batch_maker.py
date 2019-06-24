from vgg19 import VGG19model
import tensorflow as tf
from path_set import *
from data_reader import DataReader
from data_dealer import DataDealer
import gensim
import os

# 屏蔽通知信息和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BatchMaker:
    
    def __init__(self):
        # 获取字典
        self.dealer = DataDealer(ANSWERS_DICT_PATH)
        # 获取样本集信息
        self.reader = DataReader(TRAIN_DATA_TYPE)
        self.reader.set_pos()
        self.weight_vgg = None
        self.biase_vgg = None
        self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_WIKI_GENSIM_DATA_PATH)

    def __getOnehot(self,pos,value,len_):
        """
        获取onehot标签
        位置为pos[i]的值为value[i]
        其他为0
        pos和value的长度必须相等
        返回列表，长度为len_
        """
        if type(pos) != list:
            pos = [pos]
        result = []
        for i in range(len_):
            if i in pos:
                result.append(value)
            else:
                result.append(0)
        return result

    def __getVGG19Result(self,img_batch):
        """
        获取VGG19网络的结果
        img_batch:tf.Variable->[1,224,224,3]
        返回最后一层隐藏层的一维列表
        """
        # 导入VGG19模型
        model = VGG19model()
        if self.weight_vgg == None and self.biase_vgg == None:
            self.weight_vgg,self.biase_vgg = model.loadWeightsAndBiases(VGG19_WEIGHTS_AND_BIASE_PATH,False)
        # 这里输出的是最后一个隐藏层
        out = model.getNet(img_batch,self.weight_vgg,self.biase_vgg,0.2,True)
        init = tf.global_variables_initializer() 
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(init)
            data = sess.run(out)
        return data[0]

    def makeTrainBatch(self,start_pos = 0,end_pos = 0):
        """
        制作训练集
        """
        # 保存位置TRAIN_BATCH_PATH
        # 速度很慢...
        if os.path.exists(os.path.dirname(TRAIN_BATCH_PATH)) == False:
            os.mkdir(os.path.dirname(TRAIN_BATCH_PATH))
        path = TRAIN_BATCH_PATH.split('.')
        writer = tf.python_io.TFRecordWriter(path[0] + str(start_pos) + '_' + str(end_pos) + '.' + path[1])
        self.reader.set_pos(start_pos)
        # 导入VGG19模型
        model = VGG19model()
        # img_batch为占位符
        img_batch = tf.placeholder(dtype = tf.float32, shape = [224,224,3], name = 'IMG')
        img_batch_1 = tf.reshape(img_batch,[1,224,224,3])
        weight_vgg,biase_vgg = model.loadWeightsAndBiases(VGG19_WEIGHTS_AND_BIASE_PATH,False)
        # 这里输出的是最后的池化层
        out = model.getNet(img_batch_1,weight_vgg,biase_vgg,0,True)
        out = tf.reshape(out,[1,7*7*512])
        init = tf.global_variables_initializer() 
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(init)
            print('running...')
            while self.reader.get_pos() < end_pos:
                now_id = self.reader.get_next_pic_id()            
                img = self.reader.get_pic_data(now_id)
                # 是否为黑白图判定
                if len(img.shape) == 2:
                    continue
                img_data = sess.run(out,feed_dict={img_batch:img})
                img_data = img_data.tolist()[0]
                qa = self.reader.get_pic_qa(now_id)
                for q in qa:
                    question = q['question']
                    question = question.replace('?','')
                    question = question.replace(',',' ,')
                    question = question.replace('.',' .')
                    question = question.split(' ')
                    answers = dict()
                    confidences = []
                    for a in q['answers']:
                        # 判断条件->对该回答的信心程度，'yes'加权1，'maybe'加权0.5，三分以上为yes
                        answer = a['answer']
                        weight = 0
                        if a['answer_confidence'] == 'yes':
                            weight = 1
                        elif a['answer_confidence'] == 'maybe':
                            weight = 0.5
                        if not answer in answers.keys():
                            answers[answer] = 0
                        answers[answer] = answers[answer] + weight
                    answers_list = []
                    for key in answers.keys():
                        if answers[key] >= 3:
                            answers_list.append(self.dealer.deal(key)[1])
                    # 若这个问题没有正确回答，则跳过
                    if len(answers_list) == 0:
                        continue
                    label = self.__getOnehot(answers_list,1,self.dealer.getWordNum())
                    data = []
                    for word in question:
                        # data.shape = (len(question)*300)
                        try:
                            data = data + list(self.model_word2vec[word])
                        except:
                            # 识别不出的填0
                            # 例如问号
                            data = data + [0] * 300
                        else:
                            pass
                    data = data + [0] * ((QUESTION_MAX_LEN - len(question))*300)
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "answer": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),# len = self.dealer.getWordNum()
                                "question": tf.train.Feature(float_list=tf.train.FloatList(value=data)),# len = len(question)*300
                                'img': tf.train.Feature(float_list=tf.train.FloatList(value=img_data)),# len = 7*7*512
                                'question_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(question)]))# len = 1
                            }
                        )
                    )
                    writer.write(example.SerializeToString())
            writer.close()
            print('over!')

if __name__ == '__main__':
    maker = BatchMaker()
    for i in range(81,82,1):
        maker.makeTrainBatch(i*1000,i*1000+1000)
        # 重新设置图
        tf.reset_default_graph()