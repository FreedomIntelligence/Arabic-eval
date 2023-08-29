from __future__ import unicode_literals
import sys
import uuid
import requests
import hashlib
import time
import os
from importlib import reload
import jsonlines
import pandas as pd
from tqdm import *
reload(sys)
import copy
import json
import multiprocessing as mp
import os
import ssl
import sys

from gpt import GPT
'''
如果没有这个包：
pip install git+https://github.com/FreedomIntelligence/GPT.git
'''
YOUDAO_URL = 'https://openapi.youdao.com/v2/api'
APP_KEY = '12a26fe7221551a3'
APP_SECRET = 'o8QZOrfTI5nTAkDrW1n43EjzCSLbXfLa'
data = 0
lang = 'Chinese'
mul = 5
blo = True
path = "llama-v2.1-scoring/"
# def connect(idx,answer):
#     text = answer[idx]
#     prompt = "Help me translate the original text to {} following " \
#              "(1) output the result directly without additional information; " \
#              "(2) do not output any prefix, e.g., `The following is the translation`;" \
#              "(3) translate according to the context to confirm the translated texts make sense; " \
#              "(4) do not modify the format, e.g., the markdown format; " \
#              "(5) Titles of references or books are not required to be translated. \n\n" \
#              "The original text is:\n\n{}"
#     model = GPT(user_name='songdj_arabic_trans', new_version='0.1.0')
#     # print(prompt.format(lang, text))
#     flag, trans_text = model.call(prompt.format(lang, text))
#     # print(flag, trans_text)
#     if not flag:
#         raise ValueError(f"Failed: {trans_text}")
#     return trans_text
def connect():
    # text = answer[idx]
    t = "please translate the whole following sentence into Arabic:\nPlease select {neutral,contradiction,entailment} which  relation is processed by sentence1 and sentence2.Please answer the question concisely, without explanation or additional information."
    prompt = "Help me translate the original text to {} following " \
             "(1) output the result directly without additional information; " \
             "(2) do not output any prefix, e.g., `The following is the translation`;" \
             "(3) translate according to the context to confirm the translated texts make sense; " \
             "(4) do not modify the format, e.g., the markdown format; " \
             "(5) Titles of references or books are not required to be translated. \n\n" \
             "The original text is:\n\n{}"
    model = GPT(user_name='songdj_arabic_trans', new_version='0.1.0')
    # print(prompt.format(lang, text))
    flag, trans_text = model.call(t)
    # print(flag, trans_text)
    if not flag:
        raise ValueError(f"Failed: {trans_text}")
    return trans_text

def getqArray():#get instruction from the file
    path1 = path+"combine.jsonl"
    path2 = path+"output0.jsonl"
    data1 = pd.read_json(path1,lines = True)
    data2 = pd.read_json(path2,lines = True)
    answer1 = []
    answer2 = []
    assistant1 = []
    assistant2 = []
    query = []
    output = []
    label = []
    for i in range(len(data1)):
        query.append(data1["question"][i])
        answer1.append(data1["answer1"][i])
        answer2.append(data1["answer2"][i])
        assistant1.append(data2['assistant1_name'][i])
        assistant2.append(data2['assistant2_name'][i])
        output.append(data2['eval_result'][i])
        label.append(data1['category'][i])

    return answer1,answer2,assistant1,assistant2,output,query,label


def get_jsonl_classification(i):#put results in a json 
    answer1,answer2,assistant1,assistant2,output,query,label= getqArray()
    data = pd.read_json(path+"classification_cot.jsonl",lines = True)
    print(len(data))
    if len(data)==0 or (len(data)>0 and i not in list(data["id"])):
        with jsonlines.open(path+"classification_cot.jsonl",'a') as writer:
            query_trans = connect(i,query)
            trans_answer1 = connect(i,answer1)
            trans_answer2 = connect(i,answer2)
            da = {}
            da["id"] = int(i)
            da['label']=label[i]
            da["query"] = query_trans
            da["chatgpt"] = trans_answer1
            da["ourmodel"] = trans_answer2
            s = output[i].split('\n')[0]
            s = output[i].split('\n')[-1]
            if s =='Assistant 1 is worse than Assistant 2':
                da['winner'] = assistant2[i]
            elif s =='Assistant 1 is better than Assistant 2':
                da['winner'] = assistant1[i]
            else:
                da["winner"] = "equal"
            writer.write(da)

def get_jsonl_scoring(i):#put results in a json 
    answer1,answer2,assistant1,assistant2,output,query,label= getqArray()
    data = pd.read_json(path+"scoring.jsonl",lines = True)
    print(len(data))
    if len(data)==0 or (len(data)>0 and i not in list(data["id"])):
        with jsonlines.open(path+"scoring.jsonl",'a') as writer:
            query_trans = connect(i,query)
            trans_answer1 = connect(i,answer1)
            trans_answer2 = connect(i,answer2)
            da = {}
            da["id"] = int(i)
            da['label']=label[i]
            da["query"] = query_trans
            da["chatgpt"] = trans_answer1
            da["ourmodel"] = trans_answer2
            s = output[i].split('\n')[0]
            s = s.split(' ')
            score1 = s[0]
            score2 = s[1]
            da[f'{assistant1[i]} score'] = score1
            da[f'{assistant2[i]} score'] = score2
            writer.write(da)

def scoring():
    result = 0
    if not os.path.exists(path+"scoring.jsonl"):
        with jsonlines.open(path+"scoring.jsonl",'a') as writer:
            da = ""
    data = []
    with mp.Pool(processes=mul) as pool:
        pool.map(get_jsonl_scoring,[i for i in (range(len(answer1)))])
    return "scoring.jsonl","sorted_scoring.jsonl"


def classification_cot():
    result = 0
    if not os.path.exists(path+"classification_cot.jsonl"):
        with jsonlines.open(path+"classification_cot.jsonl",'a') as writer:
            da = ""
    data = []
    with mp.Pool(processes=mul) as pool:
        pool.map(get_jsonl_classification,[i for i in (range(len(answer1)))])
    return "classification_cot.jsonl","sorted_classification_cot.jsonl"


def sortup(file1,file2):
    data = pd.read_json(path+file1,lines = True,encoding = 'utf-8')
    data.sort_values(by="id" , inplace=True, ascending=True)
    sort_data = json.loads(data.to_json(orient='records',force_ascii=False))
    with jsonlines.open(path+file2,'w') as writer:
        for j in sort_data:
            writer.write(j)
    create_json(file2)
    print("Finished!")

def create_json(file2):
    data = pd.read_json(path+file2,lines = True,encoding = 'utf-8')
    data.sort_values(by="id" , inplace=True, ascending=True)
    sort_data = json.loads(data.to_json(orient='records',force_ascii=False))
    print(sort_data)
    data = []
    for j in sort_data:
        data.append(j)
    with open(path+file2.split('.')[0]+".json",'w',encoding = 'utf-8') as file:
        json.dump(data,file,ensure_ascii=False)
    print("Finished!")
 
if __name__ == '__main__':
    # answer1,answer2,assistant1,assistant2,output,query,label= getqArray()
    # #要运行程序，需要修改处：
    # #1.修改main中的path1，path1路径下存放两个jsonl文件：combine和outputs0
    # #2.若要处理classfication_cot结果，运行classification_cot(),处理scoring结果，运行scoring()
    # # scoring()
    # # sortup_scoring()
    # # string1,string2=scoring()
    # # sortup(string1,string2)
    print(connect())
