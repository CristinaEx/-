import gensim
import os
from path_set import *

def make_model_from_glove_data(data_path,save_file_path):
    """
    制作数据集
    在开头加上一行注明 向量数 向量维度
    """
    with open(data_path,'r',encoding='utf-8') as f:
        data = f.readlines()
        word_num = len(data)
        line = ''
        for d in data[0]:
            line = line + d
        line = line.split(' ')
        word_div = len(line) - 1
    with open(save_file_path, 'w+',encoding='utf-8') as f:
        f.write(str(word_num) + ' ' + str(word_div) + '\n')
        for d in data:
            line = ''
            for word in d:
                line = line + word
            f.write(line)


if __name__ == '__main__':
    # make_model_from_glove_data(GLOVE_WIKI_DATA_PATH,GLOVE_WIKI_GENSIM_DATA_PATH)
    model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_WIKI_GENSIM_DATA_PATH)
    print(model['love'])