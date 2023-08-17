# Data preprocessing
import pandas as pd
import jsonlines
import os
address1 = "ALUE-main/Cross-lingual Sentence Representations/arabic_dev.tsv"
address2 = "ALUE-main/NSURL-2019 Shared Task 8 (Question to Question Semantic Similarity)/train.csv"
address3 = "ALUE-main/OSACT4 Shared Task-A offensive/OSACT2020-sharedTask-dev.txt"
address4 = "ALUE-main/SemEval-2018 Task 1 - Affect in Tweets Emotion Classification task/2018-E-c-Ar-dev.txt"
address5 = "ALUE-main/SemEval-2018 Task 1 - Affect in Tweets Sentiment Intensity Regression task/2018-Valence-reg-Ar-dev.txt"
address6 = "vicuna80+20-en/en100.jsonl"
address7 = "MMLU/data/dev/"
address8 = "MMLU/MMLU-gpt4.jsonl"
address9 = "MMLU/mmlu_Arabic/dev/"
address10 = "path.jsonl"
address11 = "vicuna-100/output-gpt4.jsonl"
address12 = "vicuna-100/vicuna100gpt3.5.jsonl"

def CrossLingual_SentenceRepresentations(address):
    data = pd.read_csv(address,sep='\t')
    with jsonlines.open('ALUE_processed/SentenceRepresentations_dev.jsonl', mode='w') as writer:
        for i in range(len(data)):
            json_data = {}
            json_data["id"]=str(i+1)
            json_data["instruction"]=data["sentence2"][i]+' & '+ data["sentence1"][i]
            json_data["input"]=""
            json_data["output"]=data["gold_label"][i]
            writer.write(json_data)


def Q2QSemanticSimilarity(address):
    data = pd.read_csv(address,sep=',')
    with jsonlines.open('ALUE_processed/Q2QSemanticSimilarity_dev.jsonl', mode='w') as writer:
        for i in range(len(data)):
            json_data = {}
            json_data["id"]=str(i+1)
            json_data["instruction"]=data["question1"][i]+' & '+ data["question2"][i]
            json_data["input"]=""
            if data["label"][i]==0:
                label = 'True'
            else:
                label = 'False'
            json_data["output"]=label
            writer.write(json_data)


def Offensive(address):
    data = pd.read_table(address,sep='\t')
    with jsonlines.open('ALUE_processed/offensive_dev.jsonl', mode='w') as writer:
        for i in range(len(data)):
            json_data = {}
            json_data["id"]=str(i+1)
            json_data["instruction"]=data["sentence"][i]
            json_data["input"]=""
            json_data["output"]= data["offensive"][i]+" "+data["hate"][i]
            writer.write(json_data)


def EmotionClassification(address):
    data = pd.read_table(address,sep='\t')
    with jsonlines.open('ALUE_processed/emotion_dev.jsonl', mode='w') as writer:
        for i in range(len(data)):
            json_data = {}
            json_data["id"]=str(i+1)
            json_data["instruction"]=data["Tweet"][i]
            json_data["input"]=""
            json_data["anger"] = str(data["anger"][i])
            json_data["anticipation"] = str(data["anticipation"][i])
            json_data["disgust"] = str(data["disgust"][i])
            json_data["fear"] = str(data["fear"][i])
            json_data["joy"] = str(data["joy"][i])
            json_data["love"] = str(data["love"][i])
            json_data["optimism"] = str(data["optimism"][i])
            json_data["pessimism"] = str(data["pessimism"][i])
            json_data["sadness"] = str(data["sadness"][i])
            json_data["surprise"] = str(data["surprise"][i])
            json_data["trust"]= str(data["trust"][i])
            writer.write(json_data)


def SentimentIntensityRegression(address):
    data = pd.read_table(address,sep='\t')
    with jsonlines.open('ALUE_processed/vincuna100-en.jsonl', mode='w') as writer:
        for i in range(len(data)):
            json_data = {}
            json_data["id"]=str(i+1)
            json_data["instruction"]=data["Tweet"][i]
            json_data["input"]=""
            json_data["Affect Dimension"]= data["Affect Dimension"][i]
            json_data["output"] = str(data["Intensity Score"][i])
            writer.write(json_data)



def ProcessVicuna100(address):
    data = pd.read_json(address,lines = True)
    with jsonlines.open('vicuna-100/vincuna100.jsonl', mode='w') as writer:
        for i in range(len(data)):
            json_data = {}
            json_data["id"]=str(i+1)
            json_data["instruction"]='Please translate the following sentences into Arabic:' + data["query"][i]
            json_data["input"]=""
            json_data["output"] = ""
            writer.write(json_data)

def ProcessMmluArabic(address):#merge all mmlu_Arabic dev datasets
    #data = pd.read_json(address1,lines = True)
    files=os.listdir(address)
    id = 0
    num = 0
    for file in files:
        path = os.path.join(address+file)
        data = pd.read_csv(path,sep=',',names = ['sentence','A','B','C','D','answer'])
        with jsonlines.open('path.jsonl', mode='a') as writer:
            for i in range(len(data)):
                id+=1
                json_data = {}
                json_data["id"]=str(id)
                json_data["instruction"]=data["sentence"][i]
                json_data["input"]=""
                json_data["output"]=""
                writer.write(json_data)


def ProcessMMLU(address):
    files=os.listdir(address)
    id = 0
    for file in files:
        path = os.path.join(address+file)
        data = pd.read_csv(path,sep=',',names = ['sentence','A','B','C','D','answer'])
        with jsonlines.open('path.jsonl', mode='a') as writer:
            for i in range(len(data)):
                id+=1
                json_data = {}
                json_data["id"]=str(id)
                json_data["instruction"]='Please translate the following sentence into Arabic:' + data["sentence"][i]
                json_data["input"]=""
                json_data["output"]=""
                writer.write(json_data)
def Process3_4(address1,address2):
    data1 = pd.read_json("vicuna-100/output-gpt4.jsonl",lines = True)
    data2 = pd.read_json(address2,lines = True)
    id = 0
    with jsonlines.open('vicuna-100/merge_vicuna.jsonl', mode='w') as writer:
        for i in range(len(data1)):
            id+=1
            json_data = {}
            json_data['id']=str(id)
            json_data ["instruction"]=data1["instruction"][i]
            json_data["output-gpt4"]=data1["output"][i]
            json_data["output-gpt3.5"]=data2["instruction"][i]
            writer.write(json_data)
def gotdata(address):
    # data = pd.read_json(address,lines = True)
    data = pd.read_json("vicuna-100/output-gpt4.jsonl",lines = True)
    id = 0
    with jsonlines.open('vicuna-100/eval_vicuna100.jsonl', mode='w') as writer:
        for i in range(len(data)):
            da = {}
            da["query_id"] = str(id)
            da["query"] = data["output"][i]
            id+=1
            writer.write(da)

def gotdata(address):
    # data = pd.read_json(address,lines = True)
    data = pd.read_json("vicuna-100/output-gpt4.jsonl",lines = True)
    data1 = pd.read_json("vicuna-100/vicuna100gpt3.5.jsonl",lines = True)
    id = 0
    with jsonlines.open('vicuna-100/eval_vicuna100.jsonl', mode='w') as writer:
        for i in range(len(data)):
            da = {}
            da["id"] = id+1
            da["label"]= data1["label"][i]
            da["query"] = data["output"][i]
            id+=1
            writer.write(da)
def gotdatas():
    # data = pd.read_json(address,lines = True)
    data = pd.read_json("vicuna-mosenchecked.json",lines = True)
    data1 = pd.read_json("vicuna_check.json",lines = True)
    id = 0
    with jsonlines.open('vicuna1.jsonl', mode='w') as writer:
        for i in range(90):
            id = id+1
            flag = True
            if id==40 and  flag:
                flag = False
                k = id+1
                for i in range(10):
                    print(i)
                    da = {}
                    da["id"] = k
                    da["label"]= ""
                    da["query"] = data["query"][i]
                    k+=1
                    writer.write(da)
            else:
                da = {}
                da["id"] = int(data1["id"][i])
                da["label"]= data1["label"][i]
                da["query"] = data1["query"][i]
                writer.write(da)
# Process3_4(address11,address12)
# # data1 = pd.read_json("vicuna-100/output-gpt4.jsonl",lines = True)
# gotdata("vicuna-100/output-gpt4.jsonl")
def gotdata1():
    # data = pd.read_json(address,lines = True)
    data = pd.read_json("vicuna1.jsonl",lines = True)
    id = 0
    with jsonlines.open('vicuna1_gpt4.jsonl', mode='w') as writer:
        for i in range(len(data)):
            id+=1
            da = {}
            da["query_id"] = id
            da["instruction"] = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.Please to answer following question with Arabic.This is the question:"+data["query"][i]
            da["input"]=""
            da["output"]=""
            writer.write(da)
def temdata(address):
    files=os.listdir(address)
    files.sort(key=lambda x:int(x.split('.')[0]))
    id = 0
    for i in range(80):
        path = os.path.join(address+files[i])
        print(path)
        data = pd.read_json(path,lines = True)
        data1 = pd.read_json("vicuna2_gpt4.jsonl",lines = True)
        with jsonlines.open('vicuna80_noprompt/merge_new.jsonl', mode='a') as writer:
            json_data = {}
            json_data["id"]=int(data["id"][0])
            json_data["label"] = data1["label"][i]
            json_data["query"]= data["instruction"][0]
            json_data["input"]=""
            json_data["output"]=data["output"][0]
            writer.write(json_data)

def pro_gpt_output(address):
    data = pd.read_json("merge.jsonl",lines = True)
    data1 = pd.read_json(address,lines = True)
    id = 0
    with jsonlines.open('vicuna1_gpt4_result.jsonl', mode='w') as writer:
        for i in range(len(data)):
            id+=1
            da = {}
            da["id"]=int(data["id"][i])
            da["label"] = data1["label"][i]
            da["query"] = data1["query"][i]
            da["output"] = data["output"][i]
            writer.write(da)
def pro_gpt_output():
    data = pd.read_json("vicuna1.jsonl",lines = True)
    id = 0
    with jsonlines.open('vicuna2_gpt4.jsonl', mode='w') as writer:
        for i in range(len(data)):
            id+=1
            da = {}
            da["id"]=int(data["query_id"][i])
            da["label"] = data["label"][i]
            da["instruction"] = data["instruction"][i]
            da["input"] = ""
            da["output"] = ""
            writer.write(da)
temdata("vicuna80_noprompt/")