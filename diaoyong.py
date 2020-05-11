# -*- coding: utf-8 -*-
# @Author  : 王小易 / SummerYee
# @Time    : 2020/4/21 15:00
# @File    : diaoyong.py
# @Software: PyCharm

from main import query_label
import time
import MySQLdb


conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='123456',
        db ='django_db1',
        )
cur = conn.cursor()
cur.execute('drop table grad_db')
char="""CREATE TABLE grad_db (
id INT(11) NOT NULL AUTO_INCREMENT PRIMARY KEY,
sentence VARCHAR ( 255 ) NOT NULL,
result VARCHAR ( 255 ) NOT NULL);
 """
cur.execute(char)
a='0'
while True:
    time.sleep(1)
    f=open('linshi.txt',encoding='utf-8')
    b=f.read()
    f.close()
    if a != b:
        # print('执行替换')
        result=query_label(b)
        a=b
        f2 = open('linshi2.txt', 'w',encoding='utf-8')
        f2.write(result)
        f2.close()