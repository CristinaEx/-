### 进行数据的预处理
import json
import io
import os
import re

# 对句子进行拆分，单词索引标记等工作
# One-Hot标签的预处理
class DataDealer:
    
    def __init__(self,path):
        """
        path:文件路径和完整的文件名
        """
        self.path = path
        if not os.path.exists(self.path):
            self.load_dict = {"WORD_NUM":2,"START_WORD":0,"END_WORD":1}
        else:     
            with open(self.path,'r') as load_f:
                try:
                    self.load_dict = json.load(load_f)
                except:
                    self.load_dict = {"WORD_NUM":0}
                else:
                    pass
        
    def saveData(self,path_ = None):
        """
        保存当前字典记录的数据
        path_:需要保存的位置，若为None则保存为打开的位置
        """
        if path_ == None:
            path_ = self.path
        with open(path_,'w+') as f:
            f.write(json.dumps(self.load_dict))

    def deal(self,sentence):
        """
        sentence 为句子
        return 处理好的句子（列表,0开头,1结束）
        """
        # 取小写
        sentence = sentence.lower()
        sentence = re.sub('[^a-z0-9]',' ',sentence)
        words = sentence.split()
        result = [0]
        for word in words:
            if word in self.load_dict.keys():
                result.append(self.load_dict[word])
            else:
                self.load_dict[word] = self.load_dict['WORD_NUM']
                result.append(self.load_dict[word])
                self.load_dict['WORD_NUM'] = self.load_dict['WORD_NUM'] + 1
        result.append(1)
        return result

    def getWordNum(self):
        """
        获取当前文本集文字数量
        """
        return self.load_dict['WORD_NUM']

    def getWord(self,index):
        """
        获取序号为index单词
        """
        for key in self.load_dict.keys():
            if self.load_dict[key] == index:
                return key

if __name__ == '__main__':
    dealer = DataDealer("dict\\answers_dict.json")
    print(dealer.deal('no'))