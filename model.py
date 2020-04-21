# -*- coding: utf-8 -*-
# @Author  : 王小易 / SummerYee
# @Time    : 2020/4/17 18:35
# @File    : model.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import collections
import jieba
import pickle
from keras.preprocessing.sequence import pad_sequences  #序列预处理 序列填充
from keras.utils import to_categorical,plot_model   # 将类别向量转换为二进制（只有0和1）的矩阵类型表示
from keras.models import Sequential   # 序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠
from keras.layers import Embedding, LSTM, Dense
from keras import backend as K
from keras.callbacks import TensorBoard   # TensorBoard是一个可视化工具，它可以用来展示网络图、张量的指标变化、张量的分布情况等
import time
from sklearn.metrics import classification_report
# from IPython.display import SVG
# from keras.utils.visualize_util import model_to_dot



def get_json_data(path):
    # 读取数据
    data_df = pd.read_json(path)
    # 转置
    data_df = data_df.transpose()
    # 改名称
    data_df = data_df[['query', 'label']]
    return data_df

train_data_df = get_json_data(path="train.json")
test_data_df = get_json_data(path="dev.json")

# print(train_data_df.head())


# 获取所有标签，也就是分类的类别
labels = ['website', 'tvchannel', 'lottery', 'chat', 'match',
          'datetime', 'weather', 'bus', 'novel', 'video', 'riddle',
          'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music',
          'cookbook', 'stock', 'map', 'message', 'poetry', 'cinemas', 'news',
          'flight', 'translation', 'train', 'schedule', 'radio', 'email']

label_numbers = len(labels)
# print(label_numbers)

# 标签和对应ID的映射字典
label_index_dict = dict([(label, index) for index, label in enumerate(labels)])
index_label_dict = dict([(index, label) for index, label in enumerate(labels)])

with open('index_label_dict.pkl', 'wb') as fo:     # 将数据写入pkl文件
    pickle.dump(index_label_dict, fo)

# 结巴分词 对元数据进行处理

# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认精确模式
# print(list(seg_list))

# 序列化

def use_jieba_cut(one_sentence):
    return list(jieba.cut(one_sentence))

train_data_df['cut_query'] = train_data_df['query'].apply(use_jieba_cut)
test_data_df['cut_query'] = test_data_df['query'].apply(use_jieba_cut)

# print(train_data_df.head(10))


# 获取数据的所有词汇

def get_all_vocabulary(data, colunm_name):
    train_vocabulary_list = []
    max_cut_query_lenth = 0

    for cut_query in data[colunm_name]:
        if len(cut_query) > max_cut_query_lenth:
            max_cut_query_lenth = len(cut_query)
        train_vocabulary_list += cut_query
    return train_vocabulary_list, max_cut_query_lenth

train_vocabulary_list, max_cut_query_lenth = get_all_vocabulary(train_data_df, 'cut_query')
# print('Number of words:', len(train_vocabulary_list))

test_vocabulary_list, test_max_cut_query_lenth = get_all_vocabulary(train_data_df, 'cut_query')
# print('Test_max_cut_query_lenth:', test_max_cut_query_lenth)

train_vocabulary_counter = collections.Counter(train_vocabulary_list)
# print('Number of different words:', len(train_vocabulary_counter.keys()))

# 不同种类的词汇个数，预留一个位置给不存在的词汇（不存在的词汇标记为0）

max_features = len(train_vocabulary_counter.keys()) + 1
# print(max_features)

# 统计低频词语
words_times_zero = 0
for word, words_times in train_vocabulary_counter.items():
    if words_times <= 1:
        words_times_zero += 1
# print('Word_times_zero:', words_times_zero)
# print('Wors_times_zero/all:', words_times_zero / len(train_vocabulary_counter))

# 制作词汇字典

def create_train_vocabulary_dict(train_vocabulary_counter):
    word_index, index_word = {}, {}
    index_number = 1
    for word, words_time in train_vocabulary_counter.most_common():
        word_index[word] = index_number
        index_word[index_number] = word
        index_number += 1
    return word_index, index_word

word_index_dict, index_word_dict = create_train_vocabulary_dict(train_vocabulary_counter)

with open("word_index_dict.pkl", 'wb') as fo:     # 将数据写入pkl文件
    pickle.dump(word_index_dict, fo)
# print(word_index_dict['我'], word_index_dict['。'])

# pq = 0
# for index, row in train_data_df.iteritems():
#     print(row[0], row[1], row[2])
#     pq += 1
#     if pq == 10:
#         break

# 向量化数据
def vectorize_data(data, label_index_dict, word_index_dict, max_cut_query_lenth):
    x_train = []
    y_train = []
    for index, row in data.iterrows():
        query_sentence = row[2]
        label = row[1]
        # 字典找不到的情况下用 0 填充
        x = [word_index_dict.get(w, 0) for w in query_sentence]
        y = [label_index_dict[label]]
        x_train.append(x)
        y_train.append(y)
    return (pad_sequences(x_train, maxlen=max_cut_query_lenth), pad_sequences(y_train, maxlen=1))

x_train, y_train = vectorize_data(train_data_df, label_index_dict, word_index_dict, max_cut_query_lenth)
x_test, y_test = vectorize_data(test_data_df, label_index_dict, word_index_dict, test_max_cut_query_lenth)
# print(x_train[0], y_train[0])


y_train = to_categorical(y_train, label_numbers)
y_test = to_categorical(y_test, label_numbers)
# print(x_train.shape, y_train.shape)

# 存储预处理过的数据
# print(type(x_test))
np.savez("preprocessed_data", x_train, y_train, x_test, y_test)

# 直接加载预处理的数据

use_preprocessed_data = True

if use_preprocessed_data == True:
    preprocessed_data = np.load("preprocessed_data.npz")
    x_train, y_train, x_test, y_test = preprocessed_data['arr_0'], preprocessed_data['arr_1'], preprocessed_data['arr_2'], preprocessed_data['arr_3'],

# print(x_train.shape, y_train.shape)

# 计算 F1 值的函数
def f1(y_true, y_pred):
    def recall(y_true, y_pred):  # 召回指标：仅计算召回的批量平均。计算召回率，这是一种多标签分类的指标选择了多少个相关项目
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):  # 精确度指标:仅计算精度的批量平均值。，这是用于多标签分类的指标有多少个相关的选定项目。
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# 设计模型

def creat_lstm_model(max_features, max_cut_query_lenth, label_numbers):

    model = Sequential()

    model.add(Embedding(input_dim=max_features, output_dim=32, input_length=max_cut_query_lenth))

    model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(label_numbers, activation='sigmoid'))
    # 尝试使用不同的优化器和不同的优化器配置
    model.compile(loss='categorical_crossentropy',
                  # categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
                  optimizer='adam',
                  metrics=[f1])  # 指标
    # plot_model(model, to_file='LSTM_model.png', show_shapes=True)
    # SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    return model


# 获取自定义时间格式的字符串
def get_customization_time():
    # return '2020_02_20_20_20_20' 年月日时分秒
    time_tuple = time.localtime(time.time())
    customization_time = "{}_{}_{}_{}_{}_{}".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                                                    time_tuple[4], time_tuple[5])
    return customization_time


# 训练模型

if 'max_features' not in dir():
    max_features = 2888
    # print('Not find max_features variable, use default max_features values:\t{}'.format(max_features))
if 'max_cut_query_lenth' not in dir():
    max_cut_query_lenth = 26
    # print('Not find max_cut_query_lenth, use default max_features values:\t{}'.format(max_cut_query_lenth))
if 'label_numbers' not in dir():
    label_numbers = 31
    # print('Not find label_numbers, use default max_features values:\t{}'.format(label_numbers))

model = creat_lstm_model(max_features, max_cut_query_lenth, label_numbers)

batch_size = 20 # 批次大小
epochs = 30  # 周期

# print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[TensorBoard(log_dir='../logs/{}'.format("lstm_{}".format(get_customization_time())))],
          validation_split=0.2)

json_string = model.to_json()
with open("model_json.pkl", 'wb') as fo:  # 将数据写入pkl文件
    pickle.dump(json_string, fo)

model.save_weights('my_model.h5')

# 模型评估

loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

# print('Test loss：', loss)
# print('Accuracy:', accuracy)


# 预测
y_pred_test = model.predict(x_test)
# print(y_pred_test.shape)

# One-hot

y_true = np.argmax(y_test, axis=1).tolist()
y_pred = np.argmax(y_pred_test, axis=1).tolist()

#查看分类的 准确率、召回率、F1值

print(classification_report(y_true, y_pred))

