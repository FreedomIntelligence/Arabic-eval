import json
import pandas as pd
import os
import jsonlines
from sklearn.preprocessing import LabelEncoder
from preprocess import ArabertPreprocessor
#
#
#
#to process all datasets into a new format which is suitable for ALUE competition 
#
#
#
class change_to_test(object):
    def prosec(self,path= "./data_generate/SEC_test.jsonl",df_test = pd.read_csv("./SEC/SEC_test.tsv", sep="\t"),output_path = "./predictions/E_c.tsv"):
        data = pd.read_json(path,lines=True)
        idx = data["id"]
        all_out = []
        preds = data["output"]
        labels_change = {'anger':'الغضب','anticipation':'التوقع','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
        label = {}
        for item in labels_change.items():
            label[item[1]]=item[0]
        labels = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
        labels_num = {}
        for i in range(len(labels)):
            labels_num[labels[i]]= i
        for pre in preds:
            a = pre.split(',')
            out = [0 for i in range(11)]
            for i in a:
                if i in label.keys():
                    i = label[i]
                    out[labels_num[i]]=1
            all_out.append(out)
        df_preds = pd.DataFrame(data=all_out, columns=labels,  index=df_test["ID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("prosec finished")
    def promq2q(self,path = "./data_generate/MQ2Q_test.jsonl",df_test = pd.read_csv("./MQ2Q/q2q_no_labels_v1.0.tsv", sep="\t"),output_path = "./predictions/q2q.tsv"):
        data = pd.read_json(path,lines = True)
        pre = []
        for i in range(len(data)):
            pre.append(data["output"][i])
        idx = data["id"]

        df_preds = pd.DataFrame(data=pre, columns=["prediction"], index=df_test["QuestionPairID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("promq2q finished")
    def profid(self,path = "./data_generate/FID_test.jsonl",df_test = pd.read_csv("./FID/IDAT_test_text.csv"),output_path = "./predictions/irony.tsv"):
        data = pd.read_json(path,lines = True)
        pre = data["output"].tolist()
        ind = data["id"].tolist()
        for i in range(len(ind)):
            ind[i]=ind[i]-1
        df_preds = pd.DataFrame(data=pre, columns=["prediction"], index=df_test["id"])
        df_preds.reset_index(inplace=True)

        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("profid finished")
    def promdd(self,path = "./data_generate/MDD_test.jsonl",df_test = pd.read_csv("./MDD/MADAR-Corpus-26-test.tsv", sep="\t", header=None, names=["Text", "label"]),output_path = "./predictions/madar.tsv"):
        
        data = pd.read_json(path,lines = True)
        pre = data["output"].tolist()
        ind = data["id"].tolist()
        df_preds = pd.DataFrame(data=pre)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, header=False, sep="\t")
        print("promdd finished")
    def prooold(self,path = "./data_generate/OOLD_test.jsonl",output_path = "./predictions/offensive.tsv"):
        
        data = pd.read_json(path,lines = True)
        pre = []
        for i in range(len(data)):
            pre.append(data["output"][i])
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"])
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, header=False, sep="\t")
        print("prooold finished")
    def proohsd(self,path = "./data_generate/OHSD_test.jsonl",output_path = "./predictions/hate.tsv"):
        data = pd.read_json(path,lines = True)
        pre = []
        for i in range(len(data)):
            pre.append(data["output"][i])
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"])
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, header=False, sep="\t")
        print("proohsd finished")
    def prosvreg(self,path = "./data_generate/SVREG_test.jsonl",df_test = pd.read_csv("./SVREG/VREG_test.tsv", sep="\t"),output_path = "./predictions/v_reg.tsv"):
        
        data = pd.read_json(path,lines = True)
        pre = []
        for (k,line) in enumerate(data["output"]):
                line = str(line)
                s = ""
                flag = False
                for i in line:
                    if i.isdigit():
                        s+=i
                        continue
                    if i =="." and flag == False:
                        s+=i
                        flag = True
                        continue
                    break
                label = float(s)
                pre.append(label)
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"],index=df_test["ID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("prosvreg finished")


    def proxnli(self,path = "./data_generate/XNLI_test.jsonl",df_test = pd.read_csv("./XNLI/arabic_dev.tsv", sep="\t"),output_path = "./predictions/xnli.tsv"):
        
        data = pd.read_json(path,lines = True)
        pre = []
        la = {"محايدة":"neutral","ضمنية":'entailment',"تناقض":'contradiction'}
        for i in range(len(data)):
            pre.append(la[data["output"][i]])
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"],index=df_test["pairID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("proxnli finished")


    def prodiag(self,path = "./data_generate/DIAG_test.jsonl",df_test = pd.read_csv("./DIAG/diagnostic.tsv", sep="\t"),output_path = "./predictions/diagnostic.tsv"):
        
        data = pd.read_json(path,lines = True)
        pre = []
        la = {"محايدة":"neutral","ضمنية":'entailment',"تناقض":'contradiction'}
        for i in range(len(data)):
            pre.append(la[data["output"][i]])
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"],index=df_test["pairID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("prodiag finished")
