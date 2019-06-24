from data_dealer import DataDealer
from path_set import *
import tensorflow as tf
import os
import numpy

# 屏蔽通知信息和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用VIS(VGG19)+LSTM网络
# VGG19输入的图像数据为224*224*3
class TrainNetTester:
    
    def __init__(self):
        # 获取字典
        self.dealer = DataDealer(ANSWERS_DICT_PATH)
        self.reader = DataReader()

    def test(self,batch_num = 1000):
        pass


if __name__ == '__main__': 
    tester = TrainNetTester()
    tester.test(1000)