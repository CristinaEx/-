from data_reader import DataReader

if __name__ == '__main__':
    count = 0
    reader = DataReader()
    reader.set_pos()
    start_id = reader.get_next_pic_id()
    qa = reader.get_pic_qa(start_id)
    max_len = 0
    for q in qa:
        question = q['question']
        len_ = len(question)
        if len_ > max_len:
            max_len = len_
        if len_ < 32:
            count = count + 1
    now_id = reader.get_next_pic_id()
    i = 0
    while now_id != start_id:
        qa = reader.get_pic_qa(now_id)
        for q in qa:
            question = q['question']
            len_ = len(question)
            if len_ > max_len:
                max_len = len_
            if len_ < 32:
                count = count + 1
        now_id = reader.get_next_pic_id()
        i = i + 1
        if i % 1000 == 0:
            print('*',end = '')
    # 问句的最大长度为
    print(max_len)
    # 小于32长度的问句数量为
    print(count)