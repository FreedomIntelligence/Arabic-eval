import json
import pandas as pd
import jsonlines
from sklearn.preprocessing import LabelEncoder
# from preprocess import ArabertPreprocessor
import os
#
#
#
#to process all datasets into a new format which is suitable for ALUE competition 
#
#
#

MQ2Q_answer_change = { "غير مكرر":'0', "مكرر":'1'}
FID_answer_change = { "سخرية":'0', "ليس سخرية":'1'}
OOLD_answer_change = {"غير مهين":"NOT_OFF", "مهين":"OFF"}
OHSD_answer_change = {"لا يحض على الكراهية":"NOT_HS", "خطاب كراهية":"HS"}
XNLI_answer_change = {"علاقة غير مترابطة":"neutral","علاقة مترابطة":'entailment',"علاقة متناقضة":'contradiction'}
class change_to_test(object):
    def prosec(self,path= "./data_generate/SEC_test.jsonl",df_test = pd.read_csv("./SEC/SEC_test.tsv", sep="\t"),output_path = "./predictions/E_c.tsv"):
        data = pd.read_json(path,lines=True)
        idx = data["id"]
        all_out = []
        preds = data["output"]
        # labels_change = {'anger':'الغضب','anticipation':'الترقب','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
        
        #new
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
            a = [word.strip() for word in a]
            a = list(filter(lambda x: x , a)) #drop ''
            out = [0 for i in range(11)]
            for i in a:
                if i in label.keys():
                    j = label[i]
                    out[labels_num[j]]=1
                else:
                    # hh added
                    print(f'bad predict {i}')
                    
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
            if data["output"][i] in MQ2Q_answer_change.keys():
                pre.append(MQ2Q_answer_change[data["output"][i]])
            else:
                pre.append('0')
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
        for i in range(len(pre)):
            if pre[i] in FID_answer_change.keys():
                pre[i]=FID_answer_change[pre[i]]
            else:
                pre[i]='0'
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
        labels_change = {'SFX': 'صفاقس', 'ALX': 'الإسكندرية', 'ALE': 'حلب', 'FES': 'فاس', 'TRI': 'طرابلس', 'MSA': 'العربية',
                          'CAI': 'القاهرة', 'ASW': 'أسوان', 'AMM': 'عمان', 'TUN': 'تونس', 'DOH': 'الدوحة', 'RIY': 'الرياض',
                          'ALG': 'الجزائر', 'KHA': 'الخرطوم', 'DAM': 'دمشق', 'RAB': 'الرباط', 'SAN': 'صنعاء', 'BEI': 'بيروت',
                          'JER': 'القدس', 'JED': 'جدة', 'BAS': 'البصرة', 'BEN': 'بنغازي', 'SAL': 'سل', 'MUS': 'مسقط',
                          'MOS': 'الموصل', 'BAG': 'بغداد'
                          }
        labels_change1 = {}
        for item in labels_change.items():
            labels_change1[item[1]]=item[0]
        for i in range(len(pre)):
            if pre[i] in labels_change1.keys():
                pre[i] = labels_change1[pre[i]]
            else:
                pre[i] ='SFX'
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
            if data["output"][i] in OOLD_answer_change.keys():
                pre.append(OOLD_answer_change[data["output"][i]])
            else:
                pre.append('NOT_OFF')
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
            if data["output"][i] in OHSD_answer_change.keys():
                pre.append(OHSD_answer_change[data["output"][i]])
            else:
                pre.append('NOT_HS')
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
                if s == "":
                    label = 0.0
                else:
                    label = float(s)
                pre.append(label)
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"],index=df_test["ID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("prosvreg finished")


    def proxnli(self,path = "./data_generate/XNLI_original_dev.jsonl",df_test = pd.read_csv("./XNLI/arabic_dev.tsv", sep="\t"),output_path = "./predictions/xnli.tsv"):
        
        data = pd.read_json(path,lines = True)
        pre = []
        for i in range(len(data)):
            if data["output"][i] in XNLI_answer_change.keys():
                pre.append(XNLI_answer_change[data["output"][i]])
            else:
                pre.append('entailment')
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"],index=df_test["pairID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("proxnli finished")


    def prodiag(self,path = "./data_generate/DIAG_original_dev.jsonl",df_test = pd.read_csv("./DIAG/diagnostic.tsv", sep="\t"),output_path = "./predictions/diagnostic.tsv"):
        
        data = pd.read_json(path,lines = True)
        pre = []
        
        for i in range(len(data)):
            if data["output"][i] in XNLI_answer_change.keys():
                pre.append(XNLI_answer_change[data["output"][i]])
            else:
                pre.append('entailment')
        idx = data["id"]
        df_preds = pd.DataFrame(data=pre, columns=["prediction"],index=df_test["pairID"])
        df_preds.reset_index(inplace=True)
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        df_preds.to_csv(output_path, index=False, sep="\t")
        print("prodiag finished")
