import csv
import pandas as pd
import jsonlines

def proXNLI():#处理tsv、csv原始数据
    # path1 = "XNLI/arabic_train.tsv"
    # path2 = "XNLI/arabic_train_processed.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = "There are 2 sentences,please tell me the relationship between them is neutral, contradiction or entailment:"
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = "\nsentence1:"+data[i][1]+"\n"
                sentence2 = "sentence2:"+data[i][2]+"\n"
                da = {}
                da["id"] = int(data[i][0])
                da["query"] =  prompt+sentence1+sentence2
                da["output"] = data[i][3]
                writer.write(da)

def proSVREG():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SVREG/2018-Valence-reg-Ar-train.txt"
     path2 = "SVREG/2018-Valence-reg-Ar-train-processed.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = 'Please score the sentiment intensity of the following sentence on a scale of 0 to 1:'#模板
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(1,len(data)):#遍历原始数据
            sentence = data["Tweet"][i]
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+sentence
            da["output"] = data['Intensity Score'][i]
            writer.write(da)
# proSEC("SVREG/2018-Valence-reg-Ar-train.txt","SVREG/2018-Valence-reg-Ar-train-processed.jsonl")
# data = pd.read_json("XNLI/arabic_train_processed.jsonl",lines = True)
# print(data["query"][0])
# proXNLI("XNLI/arabic_train.tsv",'XNLI/arabic_train_processed.jsonl')
proSVREG()
