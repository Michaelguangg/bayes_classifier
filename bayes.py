# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:06:07 2018

@author: guan
"""

import os
import time
import random
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


#读取停用词
def make_words_set(words_file):
    words_set = set()
    with open(words_file, 'r',encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set
#停用词写入txt
def words_set_write(words_file,words_set):
    with open(words_file, 'w',encoding='utf-8') as fp:
        for word in words_set:
            fp.write(word+'\n')
#读入文件，分词后写入中间表
def read_write(oldfile,newfile):
    with open(oldfile,'r',encoding='utf-8') as fp:   
        wd=fp.read()
        wd = wd.replace('腾讯科技', '')
        wd = wd.replace('腾讯财经', '')
        wd = wd.replace('腾讯体育', '')
        wd = wd.replace('腾讯汽车', '')
        wd = wd.replace('腾讯娱乐', '')
        wd = wd.replace('腾讯房产', '')
        wd = wd.replace('人民网', '')
        wd = wd.replace('新华网', '')
        wd = wd.replace('中新网', '')
        if len(wd)<=1:
            print(oldfile)#将空文件排除
            return
        wd_cut=jieba.cut(wd)
        wd_result=' '.join(wd_cut)
        with open(newfile, 'w',encoding='utf-8') as fw:
            fw.write(wd_result)
        fw.close()
    fp.close() 
#分词后写入中间表--训练集       
oldpath=r'\data\training'
newpath=r'\mid\train'
files=os.listdir(oldpath)
for f in files:
    oldfile=os.path.join(oldpath,f)
    newfile=os.path.join(newpath,f)
    read_write(oldfile,newfile)
#读取中间表数据--训练集
trains=[]
y_train=[]
files=os.listdir(newpath)
for f in files:
    newfile=os.path.join(newpath,f)
    with open(newfile,'r',encoding='utf-8') as fp:  
        result=fp.read()
        trains.append(result)
        y_train.append(f[0].split('_')[0])
    fp.close()
#分词后写入中间表--测试集  
oldpath=r'\data\test'
newpath=r'\mid\test'
files=os.listdir(oldpath)
for f in files:
    oldfile=os.path.join(oldpath,f)
    newfile=os.path.join(newpath,f)
    read_write(oldfile,newfile)
#读取中间表数据--测试集
tests=[]
y_test=[]
files=os.listdir(newpath)
for f in files:
    newfile=os.path.join(newpath,f)
    with open(newfile,'r',encoding='utf-8') as fp:  
        result=fp.read()
        tests.append(result)
        y_test.append(f[0].split('_')[0])
    fp.close()
#得到停用词   
stop_words=make_words_set(r'\stop_words.txt')
stop_word=list(stop_words)   
#特征处理，TF-IDF特征处理方法        
#注意，测试样本调用的是transform接口
vector = TfidfVectorizer(stop_words=stop_word)
x_train=vector.fit_transform(trains)
x_test = vector.transform(tests)

clf = MultinomialNB().fit(x_train, y_train)
clf.score(x_test,y_test) #0.82714285714285718

#找出每个文件最大的TF-IDF值及其对应的词
wordlist = vector.get_feature_names()
weightlist = x_train.toarray()
with open (r'\feature_word.txt','w',encoding='utf-8') as fp:  
    for i in range(0,6280):
        x0=y_train[i]
        x=np.where(weightlist[i]==np.max(weightlist[i]))
        x1=x[0].tolist()
        x2=np.max(weightlist[i])
        if len(x1)==1:
            x3=wordlist[x1[0]]
        else:
            x3=[]
            print(i)
            for j in range(len(x1)):
                x3.append(wordlist[x1[j]])
        xx="%s  %s  %s \n"%(str(x0),str(x2),x3)
        fp.write(xx)
fp.close()