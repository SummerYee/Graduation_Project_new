# -*- coding: utf-8 -*-
# @Author  : 王小易 / SummerYee
# @Time    : 2020/4/21 14:44
# @File    : views.py
# @Software: PyCharm


from django.shortcuts import render
from django.db import connection
import time


def index(request):
    result = '待输入'
    if request.method == 'GET':
        return render(request, 'index.html')
    else:
        sentence = request.POST.get('sentence')
        f = open('linshi.txt','w', encoding='utf-8')
        f.write(sentence)
        f.close()
        time.sleep(1)
        f2 = open('linshi2.txt', encoding='utf-8')
        result = f2.read()
        f2.close()
        cursor = connection.cursor()
        re2 = 'insert into grad_db (sentence,result) values ("{}","{}")'.format(sentence,result)
        cursor.execute(re2)
        cursor.execute('select id,sentence,result from grad_db')
        rows = cursor.fetchall()
    context = {
        'inputt': sentence,
        'result': result,
        'rows': rows
    }
    return render(request,'index.html',context=context)