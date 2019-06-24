from path_set import *
from vqaTools.vqa import VQA
import sys
import numpy
from PIL import Image
sys.path.append('D:\\VQA\\PythonHelperTools')
class DataReader:
    
    def __init__(self, data_type = TRAIN_DATA_TYPE, shape = (224,224)):
        """
        shape为输出图像数据的shape
        data_type为需导入的数据集的类型
        """
        self.data_type = data_type
        annFile='{}\\annotations\\{}{}_{}_annotations.json'.format(DATA_PATH,VERSION_TYPE,DATA_TYPE,self.data_type)
        quesFile ='{}\\Questions\\{}{}_{}_{}_questions.json'.format(DATA_PATH,VERSION_TYPE,TASK_TYPE,DATA_TYPE,self.data_type)
        self.vqa = VQA(annFile, quesFile)
        self.img_ids = list(self.vqa.imgToQA.keys())
        self.pos = 0
        self.shape = shape
        questions = self.vqa.getQuestionsFile()
        questions = questions['questions']
        # qf为通过id索引查找question的字典
        self.qf = dict()
        for q in questions:
            self.qf[q["question_id"]] = q["question"]

    def get_pic_data(self,pic_id):
        """
        获取图像数据
        pic_id:图像的id
        return numpy三维数组
        """
        imgFilename = 'COCO_' + self.data_type + '_'+ str(pic_id).zfill(12) + '.jpg'
        path = '{}\\images\\{}\\'.format(DATA_PATH,self.data_type)
        img = Image.open(path+imgFilename)
        img = img.resize(self.shape)
        return numpy.array(img)

    def get_pic_qa(self,pic_id):
        """
        获取图像的问题和回答
        return ['question_type':str,'question_id':num,'answers':[{'answer':str,'answer_confidence':'yes'|'maybe'|'no','answer_id':num}...]]
        """
        q = self.vqa.getQuesIds(imgIds=[pic_id])
        qas = self.vqa.loadQA(q)
        result = []
        for qa in qas:
            q = self.get_question(qa['question_id'])
            result.append({"question":q,"answers":qa["answers"]})
        return result

    def set_pos(self,pos = 0):
        """
        设置当前读取游标
        """
        self.pos = pos if pos < len(self.img_ids) else len(self.img_ids) + 1

    def get_pos(self):
        """
        获取当前pos
        """
        return self.pos

    def get_next_pic_id(self):
        """
        获取下一张图片的id(即当前游标所在图像的id)
        并且使索引+1
        """
        img_id = self.img_ids[self.pos]
        self.pos = self.pos + 1 if not self.pos + 1 == len(self.img_ids) else 0
        return img_id

    def get_question(self,question_id):
        return self.qf[question_id]

if __name__ == '__main__':
    reader = DataReader(TRAIN_DATA_TYPE)
    reader.set_pos(100)
    id_ = reader.get_next_pic_id()
    print(id_)
    print(reader.get_pic_qa(id_))
    