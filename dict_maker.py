# 创建字典
from path_set import *
from data_dealer import DataDealer
from data_reader import DataReader

def createAnswersDict():
    """
    创建回答字典
    """
    reader = DataReader()
    reader.set_pos()
    dealer = DataDealer(ANSWERS_DICT_PATH)
    start_id = reader.get_next_pic_id()
    qa = reader.get_pic_qa(start_id)
    for q in qa:
        answers = dict()
        for a in q['answers']:
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
                dealer.deal(key)
    now_id = reader.get_next_pic_id()
    i = 0
    while now_id != start_id:
        qa = reader.get_pic_qa(now_id)
        for q in qa:
            answers = dict()
            for a in q['answers']:
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
                    dealer.deal(key)
        now_id = reader.get_next_pic_id()
        i = i + 1
        if i % 1000 == 0:
            print('*',end = '')
    dealer.saveData()
    print('over!')

def createQuestionsDict():
    """
    创建问题字典（包含回答字典）
    """
    reader = DataReader()
    reader.set_pos()
    dealer = DataDealer(ANSWERS_DICT_PATH)
    start_id = reader.get_next_pic_id()
    qa = reader.get_pic_qa(start_id)
    for q in qa:
        question = q['question']
        dealer.deal(question)
    now_id = reader.get_next_pic_id()
    i = 0
    while now_id != start_id:
        qa = reader.get_pic_qa(now_id)
        for q in qa:
            question = q['question']
            dealer.deal(question)
        now_id = reader.get_next_pic_id()
        i = i + 1
        if i % 1000 == 0:
            print('*',end = '')
    dealer.saveData(QUESTIONS_DICT_PATH)
    print('over!')

if __name__ == '__main__':
    createAnswersDict()
    # createQuestionsDict()