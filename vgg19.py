import tensorflow as tf
import scipy.io
import scipy.misc

# VGG19网络模型
class VGG19model:
    def __init__(self):
        pass

    def loadWeightsAndBiases(self,file_path,trainable = True):
        """
        导入参数
        file_path:导入数据的位置和文件名
        trainable:是否可训练
        返回VGG19的权重和偏值
        """
        data = scipy.io.loadmat(file_path)
        weights = {  
            # 3x3 conv, 3 input, 24 outputs  
            'wc1': tf.Variable(data['layers'][0][0][0][0][0][0][0],trainable = trainable),  
            'wc2': tf.Variable(data['layers'][0][2][0][0][0][0][0],trainable = trainable),  
            # pool
            'wc3': tf.Variable(data['layers'][0][5][0][0][0][0][0],trainable = trainable),  
            'wc4': tf.Variable(data['layers'][0][7][0][0][0][0][0],trainable = trainable),  
            # pool  
            'wc5': tf.Variable(data['layers'][0][10][0][0][0][0][0],trainable = trainable),  
            'wc6': tf.Variable(data['layers'][0][12][0][0][0][0][0],trainable = trainable),  
            'wc7': tf.Variable(data['layers'][0][14][0][0][0][0][0],trainable = trainable),  
            'wc8': tf.Variable(data['layers'][0][16][0][0][0][0][0],trainable = trainable),  
            # pool
            'wc9': tf.Variable(data['layers'][0][19][0][0][0][0][0],trainable = trainable),  
            'wc10': tf.Variable(data['layers'][0][21][0][0][0][0][0],trainable = trainable),  
            'wc11': tf.Variable(data['layers'][0][23][0][0][0][0][0],trainable = trainable),  
            'wc12': tf.Variable(data['layers'][0][25][0][0][0][0][0],trainable = trainable),  
            # pool
            'wc13': tf.Variable(data['layers'][0][28][0][0][0][0][0],trainable = trainable),  
            'wc14': tf.Variable(data['layers'][0][30][0][0][0][0][0],trainable = trainable),  
            'wc15': tf.Variable(data['layers'][0][32][0][0][0][0][0],trainable = trainable),  
            'wc16': tf.Variable(data['layers'][0][34][0][0][0][0][0],trainable = trainable),  
            # pool
            # fully connected   7 7 512 4096
            'wd1': tf.Variable(data['layers'][0][37][0][0][0][0][0],trainable = trainable), 
            'wd2': tf.Variable(data['layers'][0][39][0][0][0][0][0],trainable = trainable),  
            # 4096 inputs, predict_num outputs (class prediction)  
            'out': tf.Variable(data['layers'][0][41][0][0][0][0][0],trainable = trainable)
        }
        # reshape -> 7*7*512 4096
        weights['wd1'] = tf.reshape(weights['wd1'],[25088,4096])
        weights['wd2'] = tf.reshape(weights['wd2'],[4096,4096])
        weights['out'] = tf.reshape(weights['out'],[4096,1000])
        biases = {  
            'bc1': tf.Variable(data['layers'][0][0][0][0][0][0][1],trainable = trainable),  
            'bc2': tf.Variable(data['layers'][0][2][0][0][0][0][1],trainable = trainable),  
            # pool
            'bc3': tf.Variable(data['layers'][0][5][0][0][0][0][1],trainable = trainable),  
            'bc4': tf.Variable(data['layers'][0][7][0][0][0][0][1],trainable = trainable),
            # pool
            'bc5': tf.Variable(data['layers'][0][10][0][0][0][0][1],trainable = trainable),  
            'bc6': tf.Variable(data['layers'][0][12][0][0][0][0][1],trainable = trainable),  
            'bc7': tf.Variable(data['layers'][0][14][0][0][0][0][1],trainable = trainable), 
            'bc8': tf.Variable(data['layers'][0][16][0][0][0][0][1],trainable = trainable),  
            # pool
            'bc9': tf.Variable(data['layers'][0][19][0][0][0][0][1],trainable = trainable),  
            'bc10': tf.Variable(data['layers'][0][21][0][0][0][0][1],trainable = trainable),
            'bc11': tf.Variable(data['layers'][0][23][0][0][0][0][1],trainable = trainable),  
            'bc12': tf.Variable(data['layers'][0][25][0][0][0][0][1],trainable = trainable),  
            # pool
            'bc13': tf.Variable(data['layers'][0][28][0][0][0][0][1],trainable = trainable), 
            'bc14': tf.Variable(data['layers'][0][30][0][0][0][0][1],trainable = trainable),
            'bc15': tf.Variable(data['layers'][0][32][0][0][0][0][1],trainable = trainable),
            'bc16': tf.Variable(data['layers'][0][34][0][0][0][0][1],trainable = trainable),
            # fully connected, 32*32*96 inputs, 4096 outputs  
            'bd1': tf.Variable(data['layers'][0][37][0][0][0][0][1],trainable = trainable),  
            'bd2': tf.Variable(data['layers'][0][39][0][0][0][0][1],trainable = trainable), 
            # 4096 inputs, predict_num outputs (class prediction)  
            'out': tf.Variable(data['layers'][0][41][0][0][0][0][1],trainable = trainable)  
        }
        biases['bc1'] = tf.reshape(biases['bc1'],(64,))
        biases['bc2'] = tf.reshape(biases['bc2'],(64,))
        biases['bc3'] = tf.reshape(biases['bc3'],(128,))
        biases['bc4'] = tf.reshape(biases['bc4'],(128,))
        biases['bc5'] = tf.reshape(biases['bc5'],(256,))
        biases['bc6'] = tf.reshape(biases['bc6'],(256,))
        biases['bc7'] = tf.reshape(biases['bc7'],(256,))
        biases['bc8'] = tf.reshape(biases['bc8'],(256,))
        biases['bc9'] = tf.reshape(biases['bc9'],(512,))
        biases['bc10'] = tf.reshape(biases['bc10'],(512,))
        biases['bc11'] = tf.reshape(biases['bc11'],(512,))
        biases['bc12'] = tf.reshape(biases['bc12'],(512,))
        biases['bc13'] = tf.reshape(biases['bc13'],(512,))
        biases['bc14'] = tf.reshape(biases['bc14'],(512,))
        biases['bc15'] = tf.reshape(biases['bc15'],(512,))
        biases['bc16'] = tf.reshape(biases['bc16'],(512,))
        biases['bd1'] = tf.reshape(biases['bd1'],(4096,))
        biases['bd2'] = tf.reshape(biases['bd2'],(4096,))
        biases['out'] = tf.reshape(biases['out'],(1000,))
        return weights,biases

    def conv2d(self, data, weights, biase, strides=1): 
        """
        卷积函数设置:
        data:数据
        weights:权重
        biase:偏差值
        strides:步长
        返回:
        卷积->RELU->返回值
        """ 
        # Conv2D wrapper, with bias and relu activation  
        data = tf.nn.conv2d(data, weights, strides=[1, strides, strides, 1], padding='SAME') 
        # print("test!") 
        # print(data)
        data = tf.nn.bias_add(data, biase)  
        # print(data)
        return tf.nn.relu(data)  

    def maxpool2d(self, data, k=2):  
        """
        池化:
        data:数据
        k:池化层大小(1,k,k,1)
        步长:(1,k,k,1)
        """
        # MaxPool2D wrapper  
        return tf.nn.max_pool(data, ksize=[1, k, k, 1], strides=[1, k, k, 1],  
                          padding='SAME')  

    def getNet(self, data, weights, biases, dropout, output_hiding = False):  
        """
        data:数据(pic_num,224,224,3)
        weights:权重
        biases:偏置
        dropout:防止过度拟合(float32)
        output_hiding:是否输出隐藏层(维度为[pic_num,4096])
        return out(输出层[pic_num,1000],数据含义是给每张图可能对应于的每个模型进行打分)
        """
        # Convolution Layer  
        conv1 = self.conv2d(data, weights['wc1'], biases['bc1'])  
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])  
        # Max Pooling (down-sampling)  
        pool1 = self.maxpool2d(conv2, k=2)  
        # Convolution Layer  
        conv3 = self.conv2d(pool1, weights['wc3'], biases['bc3'])  
        conv4 = self.conv2d(conv3, weights['wc4'], biases['bc4'])  
        # Max Pooling (down-sampling)  
        pool2 = self.maxpool2d(conv4, k=2)  
        # Convolution Layer  
        conv5 = self.conv2d(pool2, weights['wc5'], biases['bc5'])  
        conv6 = self.conv2d(conv5, weights['wc6'], biases['bc6'])  
        conv7 = self.conv2d(conv6, weights['wc7'], biases['bc7']) 
        conv8 = self.conv2d(conv7, weights['wc8'], biases['bc8'])  
        # Max Pooling  
        pool3 = self.maxpool2d(conv8, k=2)  
        # Convolution Layer   
        conv9 = self.conv2d(pool3, weights['wc9'], biases['bc9'])  
        conv10 = self.conv2d(conv9, weights['wc10'], biases['bc10'])   
        conv11 = self.conv2d(conv10, weights['wc11'], biases['bc11'])  
        conv12 = self.conv2d(conv11, weights['wc12'], biases['bc12'])  
        # Max Pooling  
        pool4 = self.maxpool2d(conv12, k=2) 
        conv13 = self.conv2d(pool4, weights['wc13'], biases['bc13'])  
        conv14 = self.conv2d(conv13, weights['wc14'], biases['bc14']) 
        conv15 = self.conv2d(conv14, weights['wc15'], biases['bc15']) 
        conv16 = self.conv2d(conv15, weights['wc16'], biases['bc16']) 
        # Max Pooling  
        pool5 = self.maxpool2d(conv16, k=2)  

        if output_hiding:
            return pool5

        # Fully connected layer  
        # Reshape conv2 output to fit fully connected layer input  
        fc1 = tf.reshape(pool5, [-1, 7*7*512])  
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])  
        fc1 = tf.nn.relu(fc1)  
        # Apply Dropout  
        # fc1 = tf.nn.dropout(fc1, dropout)  

        # fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])  
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])  
        fc2 = tf.nn.relu(fc2)   
        # Apply Dropout  
        # fc2 = tf.nn.dropout(fc2, dropout)  

        # Output, class prediction  
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])  
        # out = self.conv2d(fc2, weights['out'], biases['out']) 
        return out