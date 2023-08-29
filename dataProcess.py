import csv
import pandas as pd
import jsonlines
import os
import random
arabic = "يرجى الإجابة على السؤال بإيجاز، دون توضيح أو معلومات إضافية."
path_to_save = "train_all1/"
path_to_save_dev = "dev/"
path_to_save_original_dev = "original_dev/"
path_to_save_test = "test/"
def proXNLI_train_dev():#处理tsv、csv原始数据
    path1 = "XNLI/arabic_train.tsv"
    path2 = path_to_save +"XNLI_train.jsonl"
    path3 = path_to_save_dev+"XNLI_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = "رجى تحديد العلاقة {محايدة، تناقض، ضمنية} التي يتم معالجتها بواسطة الجملة 1 والجملة 2."+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = "جملة1: "+data[i][1]
                sentence2 = "جملة2: "+data[i][2]
                da = {}
                da["id"] = int(data[i][0])
                da["label"] = "XNLI"
                da["query"] =  prompt+'\n'+'\n'+sentence1+'\n'+sentence2
                da["answer"] = data[i][3]
                if i%10!=0:
                    writer.write(da)
                else:
                    dev.append(da)
                    continue
    with jsonlines.open(path3,'w') as writer:
        for i in range(len(dev)):
            writer.write(dev[i])
    print("proXNLI_train_dev finished")
proXNLI_train_dev()


def proSVREG_train_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SVREG/2018-Valence-reg-Ar-train.txt"
     path2 = path_to_save +"SVREG_train.jsonl"
     path3 = path_to_save_dev+"SVREG_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = 'يرجى تسجيل شدة المشاعر، والتي تعني شدة العاطفة في الجملة التالية على مقياس من 0 إلى 1.'+arabic#模板
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data["Tweet"][i]
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+"\n"+sentence
            da["answer"] = (str(data['Intensity Score'][i]))
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
     path3 = path_to_save_dev+"SEC_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = 'ما هي المشاعر التي يحتويها الجملة التالية؟ الرجاء الاختيار من بين هذه الخيارات: (الغضب، الترقب، الاشمئزاز، الخوف، الفرح، الحب، التفاؤل، التشاؤم، الحزن، المفاجأة، الثقة).'+arabic#模板
     labels = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
     labels_change = {'anger':'الغضب','anticipation':'التوقع','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data["Tweet"][i]
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+"\n"+sentence
            output =""
            for j in labels:
                if int(data[j][i]) == 1 :
                    output = output + labels_change[j] + ',' 
            da["answer"] = output[0:-1]
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
     path3 = path_to_save_dev+"MQ2Q_dev.jsonl"
     quotechar=None
     delimiter="\t"
     prompt = "ل تعبر هاتان الجملتان عن نفس المعنى؟ إذا كانت تعني نفس الشيء، يجب أن تكون إجابتك 1، وإلا فإنها تكون 0."+arabic
     with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(1,len(data)):#遍历原始数据
            sentence1 = data[i][0]
            sentence2 = data[i][1]
            da = {}
            da["id"] = int(i)
            da["query"] =  prompt+"\n"+"\n"+'ملة1: '+sentence1+"\n"+'ملة2: '+sentence2
            da["answer"] = data[i][2]
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
     path3 = path_to_save_dev+"FID_dev.jsonl"
     quotechar=None
     delimiter=","
     prompt = "يرجى تحديد ما إذا كان هناك أي جزء من الجملة التالية يحتوي على إيرونيا، والتي تشير إلى استخدام الكلمة للتعبير عن شيء آخر غير المعنى الحرفي وخاصة عكسه.  إذا كان يحتوي على ذلك، يجب أن تكون الإجابة 1.  وإلا، يجب أن تكون الإجابة 0."+arabic
    #     data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
     data = pd.read_csv(path1).values.tolist()
     dev = []
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data[i][1]
            da = {}
            da["id"] = int(data[i][0]+1)
            da["query"] =  prompt+"\n"+"\n"+sentence
            da["answer"] = data[i][2]
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
    path3 = path_to_save_dev+"MDD_dev.jsonl"
    data = pd.read_csv(path1,sep = '\t', header=None).values.tolist()
    dev = []
    with jsonlines.open(path2,'w') as writer: 
        for i in range(len(data)):
            da = {}
            prompt ="من فضلك قم بتحديد الجملة التالية إلى أي مدينة عربية تنتمي؟"+arabic
            da["id"] = int(i)
            da["label"] = "MDD"
            da ["query"] = prompt+'\n'+"\n"+data[i][0]
            da["answer"] = data[i][1]
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
    path3 = path_to_save_dev+"OHSD_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt =" يرجى تحديد ما إذا كانت الجملة التالية تحتوي على خطاب كراهية أم لا. إذا كانت تحتوي على خطاب كراهية، يرجى الرد بـ 'HS'، وإلا فالرد بـ 'NOT_HS'."+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["label"] = "OHSD"
                da["query"] =  prompt+"\n"+"\n"+data[i][0]
                da["answer"] = data[i][2]
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
    path3 = path_to_save_dev+"OOLD_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt ="يرجى تقييم ما إذا كانت الجملة التالية تحتوي على تعليقات مسيئة أم لا.  إذا كانت تحتوي على تعليقات مسيئة، يرجى الرد بـ 'OFF'، وإلا يرجى الرد بـ 'NOT_OFF'."+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["label"] = "OOLD"
                da["query"] =  prompt+"\n"+"\n"+data[i][0]
                da["answer"] = data[i][1]
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
    prompt = "رجى تحديد العلاقة {محايدة، تناقض، ضمنية} التي يتم معالجتها بواسطة الجملة 1 والجملة 2."+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = "جملة1: "+data[i][1]
                sentence2 = "جملة2: "+data[i][2]
                da = {}
                da["id"] = int(data[i][0])
                da["label"] = "XNLI"
                da["query"] =  prompt+'\n'+'\n'+sentence1+'\n'+sentence2
                da["answer"] = data[i][3]
                writer.write(da)
    print("proXNLI_dev finished")

def proSVREG_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SVREG/2018-Valence-reg-Ar-dev.txt"
     path2 = path_to_save_original_dev +"SVREG_original_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = 'يرجى تسجيل شدة المشاعر، والتي تعني شدة العاطفة في الجملة التالية على مقياس من 0 إلى 1.'+arabic#模板
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data["Tweet"][i]
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+"\n"+sentence
            da["answer"] = (str(data['Intensity Score'][i]))
            writer.write(da)
     print("proSVREG_dev finished")


def proSEC_dev():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SEC/2018-E-c-Ar-dev.txt"
     path2 = path_to_save_original_dev +"SEC_original_dev.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = 'ما هي المشاعر التي يحتويها الجملة التالية؟ الرجاء الاختيار من بين هذه الخيارات: (الغضب، الترقب، الاشمئزاز، الخوف، الفرح، الحب، التفاؤل، التشاؤم، الحزن، المفاجأة، الثقة).'+arabic#模板
     labels = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
     labels_change = {'anger':'الغضب','anticipation':'التوقع','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data["Tweet"][i]
            da = {}
            da["id"] = (data['ID'][i])
            da["query"] =  prompt+"\n"+"\n"+sentence
            output =""
            for j in labels:
                if int(data[j][i]) == 1 :
                    output = output + labels_change[j] + ',' 
            da["answer"] = output[0:-1]
            writer.write(da)
     print("proSEC_dev finished")


def proMDD_dev():
    path1 = "MDD/MADAR-Corpus-26-dev.tsv"
    path2 = path_to_save_original_dev +"MDD_original_dev.jsonl"
    data = pd.read_csv(path1,sep = '\t', header=None).values.tolist()
    dev = []
    with jsonlines.open(path2,'w') as writer: 
        for i in range(len(data)):
            da = {}
            prompt ="من فضلك قم بتحديد الجملة التالية إلى أي مدينة عربية تنتمي؟"+arabic
            da["id"] = int(i)
            da["label"] = "MDD"
            da ["query"] = prompt+'\n'+"\n"+data[i][0]
            da["answer"] = data[i][1]
            writer.write(da)
    print("proMDD_dev finished")


def proOHSD_dev():
    path1 = "OHSD/dev.txt"
    path2 = path_to_save_original_dev+"OHSD_original_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt =" يرجى تحديد ما إذا كانت الجملة التالية تحتوي على خطاب كراهية أم لا. إذا كانت تحتوي على خطاب كراهية، يرجى الرد بـ 'HS'، وإلا فالرد بـ 'NOT_HS'."+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["label"] = "OHSD"
                da["query"] =  prompt+"\n"+"\n"+data[i][0]
                da["answer"] = data[i][2]
                writer.write(da)
    print("proOHSD_dev finished")


def proOOLD_dev():
    path1 = "OOLD/dev.txt"
    path2 = path_to_save_original_dev+"OOLD_original_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt ="يرجى تقييم ما إذا كانت الجملة التالية تحتوي على تعليقات مسيئة أم لا.  إذا كانت تحتوي على تعليقات مسيئة، يرجى الرد بـ 'OFF'، وإلا يرجى الرد بـ 'NOT_OFF'."+arabic
    dev = []
    with jsonlines.open(path2,'w') as writer:
        for i in range(0,len(data)):
                da = {}
                da["id"] = i
                da["label"] = "OOLD"
                da["query"] =  prompt+"\n"+"\n"+data[i][0]
                da["answer"] = data[i][1]
                writer.write(da)
    print("proOOLD_dev finished")


def proDIAG_dev():#处理tsv、csv原始数据
    path1 = "DIAG/diagnostic.tsv"
    path2 = path_to_save_original_dev+"DIAG_original_dev.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = "رجى تحديد العلاقة {محايدة، تناقض، ضمنية} التي يتم معالجتها بواسطة الجملة 1 والجملة 2."+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = "جملة1: "+data[i][1]
                sentence2 = "جملة2: "+data[i][2]
                da = {}
                da["id"] = int(data[i][0])
                da["label"] = "DIAG"
                da["query"] =  prompt+'\n'+'\n'+sentence1+'\n'+sentence2
                da["answer"] = data[i][3]
                writer.write(da)
    print("proDIAG_dev finished")


def proSVREG_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SVREG/VREG_test.tsv"
     path2 = path_to_save_test +"SVREG_test.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = 'يرجى تسجيل شدة المشاعر، والتي تعني شدة العاطفة في الجملة التالية على مقياس من 0 إلى 1.'+arabic#模板
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data["Tweet"][i]
            da = {}
            da["id"] = (data['ID'][i])
            da["label"] = "SVREG"
            da["query"] =  prompt+"\n"+"\n"+sentence
            writer.write(da)
     print("proSVREG_test finished ")


def proSEC_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "SEC/SEC_test.tsv"
     path2 = path_to_save_test +"SEC_test.jsonl"
     data = pd.read_csv(path1,sep = '\t')#读入数据
     prompt = 'ما هي المشاعر التي يحتويها الجملة التالية؟ الرجاء الاختيار من بين هذه الخيارات: (الغضب، الترقب، الاشمئزاز، الخوف، الفرح، الحب، التفاؤل، التشاؤم، الحزن، المفاجأة، الثقة).'+arabic#模板
     labels = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
     labels_change = {'anger':'الغضب','anticipation':'التوقع','disgust':'الاشمئزاز','fear':'الخوف','joy':'البهجة','love':'الحب','optimism':'التفاؤل','pessimism':'التشاؤم','sadness':'الحزن','surprise':'مفاجأة','trust':'الثقة'}
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data["Tweet"][i]
            da = {}
            da["id"] = int((data['ID'][i]))
            da["label"]="SEC"
            da["query"] =  prompt+"\n"+"\n"+sentence
            writer.write(da)
     print("proSEC_test finished ")

def proMQ2Q_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "MQ2Q/q2q_no_labels_v1.0.tsv"
     path2 = path_to_save_test +"MQ2Q_test.jsonl"
     quotechar=None
     delimiter="\t"
     prompt = "ل تعبر هاتان الجملتان عن نفس المعنى؟ إذا كانت تعني نفس الشيء، يجب أن تكون إجابتك 1، وإلا فإنها تكون 0."+arabic
     with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(1,len(data)):#遍历原始数据
            sentence1 = data[i][0]
            sentence2 = data[i][1]
            da = {}
            da["id"] = int(i)
            da["label"] = "MQ2Q"
            da["query"] =  prompt+"\n"+"\n"+'ملة1: '+sentence1+"\n"+'ملة2: '+sentence2
            writer.write(da)
     print("proMQ2Q_test finished ")


def proFID_test():#path1 是原始数据文件，path2是生成的文件,处理txt原始数据
     path1 = "FID/IDAT_test_text.csv"
     path2 = path_to_save_test +"FID_test.jsonl"
     quotechar=None
     delimiter=","
     prompt = "يرجى تحديد ما إذا كان هناك أي جزء من الجملة التالية يحتوي على إيرونيا، والتي تشير إلى استخدام الكلمة للتعبير عن شيء آخر غير المعنى الحرفي وخاصة عكسه.  إذا كان يحتوي على ذلك، يجب أن تكون الإجابة 1.  وإلا، يجب أن تكون الإجابة 0."+arabic
     data = pd.read_csv(path1).values.tolist()
     with jsonlines.open(path2,'w') as writer:#打开jsonlines写入
        for i in range(len(data)):#遍历原始数据
            sentence = data[i][1]
            da = {}
            da["id"] = int(data[i][0]+1)
            da["label"] = "FID"
            da["query"] =  prompt+"\n"+"\n"+sentence
            writer.write(da)
     print("proFID_test finished ")

def proMDD_test():
    path1 = "MDD/MADAR-Corpus-26-test.tsv"
    data = pd.read_csv(path1,sep = '\t', header=None).values.tolist()
    with jsonlines.open("MDD/MDD_test.jsonl",'w') as writer: 
        for i in range(len(data)):
            da = {}
            prompt ="من فضلك قم بتحديد الجملة التالية إلى أي مدينة عربية تنتمي؟"+arabic
            da["id"] = int(i)
            da["label"] = "MDD"
            da ["query"] = prompt+'\n'+"\n"+data[i][0]
            writer.write(da)
    print("proMDD_test finished")
def proOHSD_test():
    path1 = "OHSD/tweets_v1.0.txt"
    path2 = path_to_save_test+"OHSD_test.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt =" يرجى تحديد ما إذا كانت الجملة التالية تحتوي على خطاب كراهية أم لا. إذا كانت تحتوي على خطاب كراهية، يرجى الرد بـ 'HS'، وإلا فالرد بـ 'NOT_HS'."+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(len(data)):
            da = {}
            da["id"] = i
            da["label"] = "OHSD"
            da["query"] =  prompt+data[i][0]
            writer.write(da)
    print("proOHSD_test finished")
def proOOLD_test():
    path1 = "OOLD/tweets_v1.0.txt"
    path2 = path_to_save_test+"OOLD_test.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt ="يرجى تقييم ما إذا كانت الجملة التالية تحتوي على تعليقات مسيئة أم لا.  إذا كانت تحتوي على تعليقات مسيئة، يرجى الرد بـ 'OFF'، وإلا يرجى الرد بـ 'NOT_OFF'."+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(len(data)):
            da = {}
            da["id"] = i
            da["label"] = "OOLD"
            da["query"] =  prompt+data[i][0]
            writer.write(da)
    print("proOOLD_test finished")
def proDIAG_test():#处理tsv、csv原始数据
    path1 = "DIAG/diagnostic.tsv"
    path2 = path_to_save_test+"DIAG_test.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = "رجى تحديد العلاقة {محايدة، تناقض، ضمنية} التي يتم معالجتها بواسطة الجملة 1 والجملة 2."+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = "جملة1: "+data[i][1]
                sentence2 = "جملة2: "+data[i][2]
                da = {}
                da["id"] = int(data[i][0])
                da["label"] = "DIAG"
                da["query"] =  prompt+'\n'+'\n'+sentence1+'\n'+sentence2
                writer.write(da)
    print("proDIAG_dev finished")
def proXNLI_test():#处理tsv、csv原始数据
    path1 = "XNLI/arabic_dev.tsv"
    path2 = path_to_save_test +"XNLI_test.jsonl"
    quotechar=None
    delimiter="\t"
    with open(path1, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))
    prompt = "رجى تحديد العلاقة {محايدة، تناقض، ضمنية} التي يتم معالجتها بواسطة الجملة 1 والجملة 2."+arabic
    with jsonlines.open(path2,'w') as writer:
        for i in range(1,len(data)):
                sentence1 = "\n"+"\n"+"جملة1: "+data[i][1]+"\n"
                sentence2 = "جملة2: "+data[i][2]
                da = {}
                da["id"] = int(data[i][0])
                da["label"] = "XNLI"
                da["query"] =  prompt+sentence1+sentence2
                writer.write(da)
    print("proXNLI_test finished")
def merge_all(path):
    dirs = os.listdir(path)
    with jsonlines.open(path+'/final.jsonl','w')as writer:
        for dir in dirs:
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
    print("merge finished")