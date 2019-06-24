# 这个项目在电脑中的位置
PROJECT_PATH = 'D:\\VQA'
# 版本
VAL_DATA_TYPE = 'val2014'
# coco_qa数据集位置
DATA_PATH = 'D:\\VQA\\coco_qa'
# 版本
VERSION_TYPE ='v2_'
# 出产
DATA_TYPE = 'mscoco'
TASK_TYPE = 'OpenEnded'
TRAIN_DATA_TYPE = 'train2014'
TRAIN_DATA_PATH = '{}\\images\\{}\\'.format(DATA_PATH,TRAIN_DATA_TYPE)
VAL_DATA_PATH = '{}\\images\\{}\\'.format(DATA_PATH,VAL_DATA_TYPE)
ANSWERS_DICT_PATH = "dict\\answers_dict.json"
QUESTIONS_DICT_PATH = "dict\\questions_dict.json"
# VGG19模型数据保存位置
VGG19_WEIGHTS_AND_BIASE_PATH = "vgg19\\imagenet-vgg-verydeep-19.mat"
# gensim模型位置
MY_GENSIM_DATA_PATH = 'word2vec\\my_word2vec.model'
GLOVE_WIKI_GENSIM_DATA_PATH = 'word2vec\\glove_wiki_word2vec.txt'
GLOVE_WIKI_DATA_PATH = 'word2vec\\glove.6B\\glove.6B.300d.txt'
# 训练集位置
TRAIN_BATCH_PATH = 'tfrecord\\train_batch.tfrecords'
# 测试集位置
TEST_BATCH_PATH = 'tfrecord\\test_batch.tfrecords'
# lstm网络的隐藏层数量
QUESTION_MAX_LEN = 36
# 模型保存位置
LSTM_MODEL_PATH = 'model\\lstm\\'
LSTM_MODEL_NAME = 'LSTM'