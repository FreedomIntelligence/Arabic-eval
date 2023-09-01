import csv
import pandas as pd
import jsonlines
import os
import random
import pyarabic.araby as araby
import html
import logging
import re
from typing import List
from preprocess import *
arabic = "يرجى الإجابة على السؤال بإيجاز، دون توضيح أو معلومات إضافية."
path_to_save = "train/"
path_to_save_dev = "dev/"
path_to_save_original_dev = "original_dev/"
path_to_save_test = "test/"
prompt_XNLI = "جى تحديد العلاقة {علاقة متناقضة,علاقة مترابطة,علاقة غير مترابطة} التي يتم معالجتها بواسطة الجملة 1 والجملة 2."
prompt_SVREG = 'يرجى تسجيل شدة المشاعر، والتي تعني شدة العاطفة في الجملة التالية على مقياس من 0 إلى 1.'
prompt_SEC = 'ما هي المشاعر التي يحتويها الجملة التالية؟ الرجاء الاختيار من بين هذه الخيارات: (الغضب, الترقب, الإشمئزاز, الخوف, الفرح, الحب, التفاؤل, التشاؤم, الحزن, المفاجئة, الثقة).'  
prompt_MQ2Q = "ل تعبر هاتان الجملتان عن نفس المعنى؟ إذا كانت تعني نفس الشيء، يجب أن تكون إجابتك مكرر، وإلا فإنها تكون غير مكرر."
prompt_DIAG = prompt_XNLI
prompt_MDD = "يرجى تحديد الجملة التالية إلى أي مدينة تنتمي لغتها، الخيارات المتاحة هي {صفاقس, الإسكندرية, حلب, فاس, طرابلس, العربية, القاهرة, أسوان, عمان, تونس, الدوحة, الرياض, الجزائر, الخرطوم, دمشق, الرباط, صنعاء, بيروت, القدس, جدة, البصرة, بنغازي, سل, مسقط, الموصل, بغداد}"
prompt_OHSD =" يرجى تحديد ما إذا كانت الجملة التالية تحتوي على خطاب كراهية أم لا. إذا كانت تحتوي على خطاب كراهية، يرجى الرد بـ 'خطاب كراهية'، وإلا فالرد بـ 'لا يحض على الكراهية'."
prompt_OOLD="يرجى تقييم ما إذا كانت الجملة التالية تحتوي على تعليقات مسيئة أم لا. إذا كانت تحتوي على تعليقات مسيئة، يرجى الرد بـ 'مهين'، وإلا يرجى الرد بـ 'غير مهين'."
prompt_FID ="يرجى تحديد ما إذا كان هناك أي جزء من الجملة التالية يحتوي على إيرونيا، والتي تشير إلى استخدام الكلمة للتعبير عن شيء آخر غير المعنى الحرفي وخاصة عكسه. إذا كان يحتوي على ذلك، يجب أن تكون الإجابةليس سخرية  وإلا، يجب أن تكون الإجابة سخرية."
XNLI_answer_change={"neutral": "علاقة غير مترابطة", "entailment": "علاقة مترابطة", "contradiction": "علاقة متناقضة"}
MDD_answer_change = {'SFX': 'صفاقس', 'ALX': 'الإسكندرية', 'ALE': 'حلب', 'FES': 'فاس', 'TRI': 'طرابلس', 'MSA': 'العربية',
                          'CAI': 'القاهرة', 'ASW': 'أسوان', 'AMM': 'عمان', 'TUN': 'تونس', 'DOH': 'الدوحة', 'RIY': 'الرياض',
                          'ALG': 'الجزائر', 'KHA': 'الخرطوم', 'DAM': 'دمشق', 'RAB': 'الرباط', 'SAN': 'صنعاء', 'BEI': 'بيروت',
                          'JER': 'القدس', 'JED': 'جدة', 'BAS': 'البصرة', 'BEN': 'بنغازي', 'SAL': 'سل', 'MUS': 'مسقط',
                          'MOS': 'الموصل', 'BAG': 'بغداد'
                          }
MQ2Q_answer_change =  {'0': "غير مكرر", '1': "مكرر"}
FID_answer_change = {'0': "سخرية", '1': "ليس سخرية"}
OOLD_answer_change = {"NOT_OFF": "غير مهين", "OFF": "مهين"}
OHSD_answer_change = {"NOT_HS": "لا يحض على الكراهية", "HS": "خطاب كراهية"}
sen1 = "جملة"+" 1: "
sen2 = "جملة"+" 2: "
def proXNLI_train_dev():#处理tsv、csv原始数据
    path1 = "XNLI/arabic_train.tsv"
    path2 = path_to_save +"XNLI_train.jsonl"
    path3 = path_to_save_dev+"XNLI_train_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_XNLI+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = preprocess_v3(data[i][1])
                sentence2 = preprocess_v3(data[i][2])
                da = {}
                da["id"] = int(data[i][0])
                da["label"] = "XNLI"
                da["query"] =  prompt+'\n'+'\n'+sen1+sentence1+'\n'+sen2+sentence2
                da["answer"] = XNLI_answer_change[data[i][3]]
                if i%10!=0:
                    writer.write(da)
                else:
                    dev.append(da)
                    continue
    with jsonlines.open(path3,'w') as writer:
        for i in range(len(dev)):
            writer.write(dev[i])
    print("proXNLI_train_dev finished")


def proSVREG_train_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SVREG/2018-Valence-reg-Ar-train.txt"
     path2 = path_to_save +"SVREG_train.jsonl"
     path3 = path_to_save_dev+"SVREG_train_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = prompt_SVREG+arabic
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data["Tweet"][i])
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+"\n"+sentence
            da["answer"] = (str(data['Intensity Score'][i]))
            da["label"] = "SVREG"
            if i%10!=0:
                writer.write(da)
            else:
                dev.append(da)
                continue
     with jsonlines.open(path3,'w') as writer:
        for i in range(len(dev)):
            writer.write(dev[i])
     print("proSVREG_train_dev finished")


def proSEC_train_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SEC/2018-E-c-Ar-train.txt"
     path2 = path_to_save +"SEC_train.jsonl"
     path3 = path_to_save_dev+"SEC_train_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = prompt_SEC+arabic
     labels = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
     labels_change = {'anger':'الغضب','anticipation':'التوقع','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data["Tweet"][i])
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] = prompt+"\n"+"\n"+sentence
            output =""
            for j in labels:
                if int(data[j][i]) == 1 :
                    output = output + labels_change[j] + ',' 
            da["answer"] = output[0:-1]
            da["label"] = "SEC"
            if i%10!=0:
                writer.write(da)
            else:
                dev.append(da)
                continue
     with jsonlines.open(path3,'w') as writer:#打开jsonlines写入
         for j in dev:
             writer.write(j)
     print("proSEC_train_dev finished")


def proMQ2Q_train_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "MQ2Q/q2q_similarity_workshop_v2.1.tsv"
     path2 = path_to_save +"MQ2Q_train.jsonl"
     path3 = path_to_save_dev+"MQ2Q_train_dev.jsonl"
     quotechar=None
     delimiter="\t"
     prompt = prompt_MQ2Q+arabic
     with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(1,len(data)):#遍历原始数据
            sentence1 = preprocess_v3(data[i][0])
            sentence2 = preprocess_v3(data[i][1])
            da = {}
            da["id"] = int(i)
            da["query"] =  prompt+"\n"+"\n"+sen1+sentence1+"\n"+sen2+sentence2
            da["answer"] = MQ2Q_answer_change[data[i][2]]
            da["label"] = "MQ2Q"
            if i%10!=0:
                writer.write(da)
            else:
                dev.append(da)
     with jsonlines.open(path3,'w') as writer:
         for j in dev:
             writer.write(j)
     print("proMQ2Q_train_dev finished")

def proFID_train_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "FID/IDAT_training_text.csv"
     path2 = path_to_save +"FID_train.jsonl"
     path3 = path_to_save_dev+"FID_train_dev.jsonl"
     quotechar=None
     delimiter=","
     prompt = prompt_FID+arabic
    #     data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
     data = pd.read_csv(path1).values.tolist()
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data[i][1])
            da = {}
            da["id"] = int(data[i][0]+1)
            da["query"] = prompt+"\n"+"\n"+sentence
            da["answer"] = FID_answer_change[str(data[i][2])]
            da["label"] = "FID"
            if i%10!=0:
                writer.write(da)
            else:
                dev.append(da)
                continue
     with jsonlines.open(path3,'w') as writer:
         for j in dev:
             writer.write(j)
     print("proFID_train_dev finished")

def proMDD_train_dev():
    path1 = "MDD/MADAR-Corpus-26-train.tsv"
    path2 = path_to_save +"MDD_train.jsonl"
    path3 = path_to_save_dev+"MDD_train_dev.jsonl"
    data = pd.read_csv(path1,sep = '\t', header=None).values.tolist()
    dev = []
    prompt = prompt_MDD+arabic
    with jsonlines.open(path2,'w') as writer: 
        for i in range(len(data)):
            da = {}
            da["id"] = int(i)
            da ["query"] = prompt+'\n'+"\n"+preprocess_v3(data[i][0])
            da["answer"] = MDD_answer_change[data[i][1]]
            da["label"] = "MDD"
            if i%10!=0:
                writer.write(da)
            else:
                dev.append(da)
                continue
    with jsonlines.open(path3,'w') as writer: 
        for j in dev:
            writer.write(j)
    print("proMDD_train_dev finished")

def proOHSD_train_dev():#处理tsv、csv原始数据
    path1 = "OOLD/OSACT2020-sharedTask-train.txt"
    path2 = path_to_save +"OHSD_train.jsonl"
    path3 = path_to_save_dev+"OHSD_train_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_OHSD+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["query"] =  prompt+"\n"+"\n"+preprocess_v3(data[i][0])
                da["answer"] = OHSD_answer_change[data[i][2]]
                da["label"] = "OHSD"
                if i %10!=0:
                    writer.write(da)
                else:
                    dev.append(da)
    with jsonlines.open(path3,'w') as writer:
        for j in dev:
            writer.write(j)
    print("proOHSD_train_dev finished")


def proOOLD_train_dev():#处理tsv、csv原始数据
    path1 = "OOLD/OSACT2020-sharedTask-train.txt"
    path2 = path_to_save +"OOLD_train.jsonl"
    path3 = path_to_save_dev+"OOLD_train_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_OOLD+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["query"] =  prompt+"\n"+"\n"+preprocess_v3(data[i][0])
                da["answer"] = OOLD_answer_change[data[i][1]]
                da["label"] = "OOLD"
                if i %10!=0:
                    writer.write(da)
                else:
                    dev.append(da)
    with jsonlines.open(path3,'w') as writer:
        for j in dev:
            writer.write(j)
    print("proOOLD_train_dev finished")
# proXNLI_train_dev()


def proXNLI_dev():#处理tsv、csv原始数据
    path1 = "XNLI/arabic_dev.tsv"
    path2 = path_to_save_original_dev +"XNLI_original_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_XNLI+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = preprocess_v3(data[i][1])
                sentence2 = preprocess_v3(data[i][2])
                da = {}
                da["id"] = int(data[i][0])
                da["query"] =  prompt+'\n'+'\n'+sen1+sentence1+'\n'+sen2+sentence2
                da["answer"] = XNLI_answer_change[data[i][3]]
                da["label"] = "XNLI"
                writer.write(da)
    print("proXNLI_dev finished")

def proSVREG_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SVREG/2018-Valence-reg-Ar-dev.txt"
     path2 = path_to_save_original_dev +"SVREG_original_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = prompt_SVREG+arabic
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data["Tweet"][i])
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+"\n"+sentence
            da["answer"] = (str(data['Intensity Score'][i]))
            da["label"] = "SVREG"
            writer.write(da)
     print("proSVREG_dev finished")


def proSEC_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SEC/2018-E-c-Ar-dev.txt"
     path2 = path_to_save_original_dev +"SEC_original_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = prompt_SEC+arabic
     labels = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
     labels_change = {'anger':'الغضب','anticipation':'الترقب','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data["Tweet"][i])
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+"\n"+sentence
            
            output =""
            for j in labels:
                if int(data[j][i]) == 1 :
                    output = output + labels_change[j] + ',' 
            da["answer"] = output[0:-1]
            da["label"] = "SEC"
            writer.write(da)
     print("proSEC_dev finished")


def proMDD_dev():
    path1 = "MDD/MADAR-Corpus-26-dev.tsv"
    path2 = path_to_save_original_dev +"MDD_original_dev.jsonl"
    data = pd.read_csv(path1,sep = '\t', header=None).values.tolist()
    dev = []
    prompt = prompt_MDD+arabic
    with jsonlines.open(path2,'w') as writer: 
        for i in range(len(data)):
            da = {}
            da["id"] = int(i)
            da ["query"] = prompt+'\n'+"\n"+preprocess_v3(data[i][0])
            da["answer"] = MDD_answer_change[data[i][1]]
            da["label"] = "MDD"
            writer.write(da)
    print("proMDD_dev finished")


def proOHSD_dev():
    path1 = "OHSD/dev.txt"
    path2 = path_to_save_original_dev+"OHSD_original_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_OHSD+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["query"] = prompt+"\n"+"\n"+preprocess_v3(data[i][0])
                da["answer"] = OHSD_answer_change[data[i][2]]
                da["label"] = "OHSD"
                writer.write(da)
    print("proOHSD_dev finished")


def proOOLD_dev():
    path1 = "OOLD/dev.txt"
    path2 = path_to_save_original_dev+"OOLD_original_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_OOLD+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["query"] =  prompt+"\n"+"\n"+preprocess_v3(data[i][0])
                da["answer"] = OOLD_answer_change[data[i][1]]
                da["label"] = "OOLD"
                writer.write(da)
    print("proOOLD_dev finished")


def proDIAG_dev():#处理tsv、csv原始数据
    path1 = "DIAG/diagnostic.tsv"
    path2 = path_to_save_original_dev+"DIAG_original_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_DIAG+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = preprocess_v3(data[i][1])
                sentence2 = preprocess_v3(data[i][2])
                da = {}
                da["id"] = int(data[i][0])
                da["query"] = prompt+'\n'+'\n'+sen1+sentence1+'\n'+sen2+sentence2
                da["answer"] = XNLI_answer_change[data[i][3]]
                da["label"] = "DIAG"
                writer.write(da)
    print("proDIAG_dev finished")


def proSVREG_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SVREG/VREG_test.tsv"
     path2 = path_to_save_test +"SVREG_test.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = prompt_SVREG+arabic
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data["Tweet"][i])
            da = {}
            da["id"] = (data['ID'][i])
            da["label"] = "SVREG"
            da["query"] = prompt+"\n"+"\n"+sentence
            writer.write(da)
     print("proSVREG_test finished ")


def proSEC_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SEC/SEC_test.tsv"
     path2 = path_to_save_test +"SEC_test.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = prompt_SEC+arabic
     labels = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
     labels_change = {'anger':'الغضب','anticipation':'التوقع','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data["Tweet"][i])
            da = {}
            da["id"] = int((data['ID'][i]))
            da["label"]="SEC"
            da["query"] = prompt+"\n"+"\n"+sentence
            writer.write(da)
     print("proSEC_test finished ")

def proMQ2Q_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "MQ2Q/q2q_no_labels_v1.0.tsv"
     path2 = path_to_save_test +"MQ2Q_test.jsonl"
     quotechar=None
     delimiter="\t"
     prompt = prompt_MQ2Q+arabic
     with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(1,len(data)):#遍历原始数据
            sentence1 = preprocess_v3(data[i][1])
            sentence2 = preprocess_v3(data[i][2])
            da = {}
            da["id"] = int(i)
            da["label"] = "MQ2Q"
            da["query"] =  prompt+"\n"+"\n"+sen1+sentence1+"\n"+sen2+sentence2
            writer.write(da)
     print("proMQ2Q_test finished ")


def proFID_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "FID/IDAT_test_text.csv"
     path2 = path_to_save_test +"FID_test.jsonl"
     quotechar=None
     delimiter=","
     prompt = prompt_FID+arabic
     data = pd.read_csv(path1).values.tolist()
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = preprocess_v3(data[i][1])
            da = {}
            da["id"] = int(data[i][0]+1)
            da["query"] = prompt+"\n"+"\n"+sentence
            da["answer"] = FID_answer_change[str(data[i][2])]
            da["label"] = "FID"
            writer.write(da)
     print("proFID_test finished ")

def proMDD_test():
    path1 = "MDD/MADAR-Corpus-26-test.tsv"
    data = pd.read_csv(path1,sep = '\t', header=None).values.tolist()
    prompt = prompt_MDD+arabic
    with jsonlines.open("test/MDD_test.jsonl",'w') as writer: 
        for i in range(len(data)):
            da = {}
            da["id"] = int(i)
            da ["query"] = prompt+'\n'+"\n"+preprocess_v3(data[i][0])
            da["answer"] = MDD_answer_change[data[i][1]]
            da["label"] = "MDD"
            writer.write(da)
    print("proMDD_test finished")
def proOHSD_test():
    path1 = "OHSD/tweets_v1.0.txt"
    path2 = path_to_save_test+"OHSD_test.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_OHSD+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(len(data)):
            da = {}
            da["id"] = i
            da["label"] = "OHSD"
            da["query"] = prompt+preprocess_v3(data[i][0])
            writer.write(da)
    print("proOHSD_test finished")
def proOOLD_test():
    path1 = "OOLD/tweets_v1.0.txt"
    path2 = path_to_save_test+"OOLD_test.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_OOLD+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(len(data)):
            da = {}
            da["id"] = i
            da["label"] = "OOLD"
            da["query"] = prompt+preprocess_v3(data[i][0])
            writer.write(da)
    print("proOOLD_test finished")
def proDIAG_test():#处理tsv、csv原始数据
    path1 = "DIAG/diagnostic.tsv"
    path2 = path_to_save_test+"DIAG_test.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = prompt_DIAG+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = preprocess_v3(data[i][1])
                sentence2 = preprocess_v3(data[i][2])
                da = {}
                da["id"] = int(data[i][0])
                da["label"] = "DIAG"
                da["query"] =  prompt+'\n'+'\n'+sen1+sentence1+'\n'+sen2+sentence2
                writer.write(da)
    print("proDIAG_dev finished")
# def proXNLI_test():#处理tsv、csv原始数据
#     path1 = "XNLI/arabic_dev.tsv"
#     path2 = path_to_save_test +"XNLI_test.jsonl"
#     quotechar=None
#     delimiter="\t"
#     with open(path1, "r", encoding="utf-8-sig") as f:
#         data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
#     prompt = prompt_XNLI+arabic
#     with jsonlines.open(path2,'w') as writer:
#         for i in range(1,len(data)):
#                 sentence1 = preprocess_v3(data[i][1])
#                 sentence2 = preprocess_v3(data[i][2])
#                 da = {}
#                 da["id"] = int(data[i][0])
#                 da["label"] = "XNLI"
#                 da["query"] = prompt+'\n'+'\n'+sen1+sentence1+'\n'+sen2+sentence2
#                 writer.write(da)
#     print("proXNLI_test finished")
def merge_all(path,path1):
    dirs = os.listdir(path)
    dirs1 = os.listdir(path1)
    with jsonlines.open('./final//final.jsonl','w')as writer:
        for dir in dirs:
            if dir!="final.jsonl":
                start = 1
                data = pd.read_json(path+'/'+dir,lines = True)
                dir_name = dir.split('.')[0]
                for i in range(len(data)):
                    da = {}
                    id = dir_name+f"-{start}"
                    da["id"] =str(id)
                    start+=1
                    da["query"] = data["query"][i]
                    da["answer"]=str(data["answer"][i])
                    writer.write(da)
        for dir in dirs1:
            if dir!="final.jsonl":
                start = 1
                data = pd.read_json(path1+'/'+dir,lines = True)
                dir_name = dir.split('.')[0]
                for i in range(len(data)):
                    da = {}
                    id = dir_name+f"-{start}"
                    da["id"] =str(id)
                    start+=1
                    da["query"] = data["query"][i]
                    da["answer"]=str(data["answer"][i])
                    writer.write(da)
    print("merge finished")