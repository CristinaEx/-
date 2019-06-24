import gensim
from data_reader import DataReader
from path_set import *
from gensim.models.keyedvectors import KeyedVectors

def data_training():
    """
    仅用样本集进行训练
    """
    sentences = []
    reader = DataReader(TRAIN_DATA_TYPE)
    reader.set_pos()
    start_id = reader.get_next_pic_id()
    qa = reader.get_pic_qa(start_id)
    for q in qa:
        question = q['question']
        question = question.replace('?',' ?')
        question = question.replace(',',' ,')
        question = question.replace('.',' .')
        sentence = question.split(' ')
        sentences.append(sentence)
    now_id = reader.get_next_pic_id()
    i = 0
    while now_id != start_id:
        qa = reader.get_pic_qa(now_id)
        for q in qa:
            question = q['question']
            question = question.replace('?',' ?')
            question = question.replace(',',' ,')
            question = question.replace('.',' .')
            sentence = question.split(' ')
            sentences.append(sentence)
        now_id = reader.get_next_pic_id()
        i = i + 1
        if i % 1000 == 0:
            print('*',end = '')
    print('load data over!')
    model = gensim.models.Word2Vec(sentences,size = 300,min_count = 1)
    model.save(GENSIM_DATA_PATH)

def get_model_from_data(file_path):
    """
    制作model，使用已经训练好的数据
    """
    model = KeyedVectors.load_word2vec_format(file_path, binary=False)  # C text format

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load(MY_GENSIM_DATA_PATH)
    # print(model.most_similar(['Latin']))
    # print(type(model['man']))
    print(model['man'])
    print(model['women'])