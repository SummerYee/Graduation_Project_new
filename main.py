# -*- coding: utf-8 -*-
# @Author  : 王小易 / SummerYee
# @Time    : 2020/3/20 23:25
# @File    : mian.py
# @Software: PyCharm

import pickle
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import jieba


# 加载 pickle 对象的函数
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# 输入模型的最终单句长度
max_cut_query_lenth = 26

# 加载查询词汇和对应 ID 的字典
word_index_dict = load_obj('word_index_dict')
# 加载模型输出 ID 和对应标签（种类）的字典
index_label_dict = load_obj('index_label_dict')
# 加载模型结构
model_json = load_obj('model_json')
model = model_from_json(model_json)
# 加载模型权重
model.load_weights('my_model.h5')


def query_label(query_sentence):
    '''
    input query: "从中山到西安的汽车。"
    return label: "bus"
    '''
    x_input = []
    # query_sentence_list = list(jieba.cut(query_sentence))
    x = [word_index_dict.get(w, 0) for w in query_sentence]
    x_input.append(x)
    x_input = pad_sequences(x_input, maxlen=max_cut_query_lenth)
    # 预测
    y_hat = model.predict(x_input)
    # 取最大值所在的序号 11
    pred_y_index = np.argmax(y_hat)
    # 查找序号所对应标签（类别）
    label = index_label_dict[pred_y_index]

    return label


if __name__ == "__main__":
    query_sentence = '红烧排骨怎么做？'
    print(query_label(query_sentence))



