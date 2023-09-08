import json
import pandas as pd
import os
import jsonlines
path1 = "data/combine.jsonl"#原始数据路径
dirs  =  os.listdir("./")
for dir in dirs:
    p = dir 
    if len(dir.split('vs'))>=2:
        break
path2 = p+"/combine.jsonl_saved.json"#输出结果路径
with open(path2,'r',encoding = 'utf-8') as read_file:
    data = json.load(read_file)
name_a = data["model_a_name"]
name_b = data["model_b_name"]
modela_name = input("请输入产生answer1的模型名字：")
modelb_name = input("请输入产生answer2的模型名字：")
path3 = modela_name + ' vs. '+modelb_name+"_processed_results"#输出结果文件夹
if not os.path.isdir(path3):
    os.mkdir(path3)
select = data["user_choices"]
num_a = 0
num_b = 0
num_draw = 0
winner = []
winner_a =[]
winner_b = []
winner_draw = []
for item in (select.items()):
    if item[1] == 'model B':
        winner.append(name_b[int(item[0])])
    else:
        winner.append(name_a[int(item[0])])
for idx,i in enumerate(winner):
    if (i=='chatgpt'):
        num_a+=1
        winner_a.append(idx)
    elif i =="draw":
        num_draw+=1
        winner_draw.append(idx)
    else:
        num_b+=1
        winner_b.append(idx)
data1 = pd.read_json(path1,lines = True).values.tolist()
with jsonlines.open(path3+f"/{modela_name}_win.jsonl",'w') as writer:
    da = {}
    da[f"{modela_name}_win_number"] = len(winner_a)
    da[f"{modelb_name}_win_number"] = len(winner_b)
    da["draw_number"] = len(winner_draw)
    writer.write(da)
    for i in winner_a:
        da = {} 
        da['id']= int(i)
        da["label"] = data1[int(i)][1]
        da[f"{modela_name}_answer"] = data1[int(i)][3]
        da[f"{modelb_name}_answer"] = data1[int(i)][4]
        writer.write(da)
with jsonlines.open(path3+f"/{modelb_name}_win.jsonl",'w') as writer:
    da = {}
    da[f"{modela_name}_win_number"] = len(winner_a)
    da[f"{modelb_name}_win_number"] = len(winner_b)
    da["draw_number"] = len(winner_draw)
    writer.write(da)
    for i in winner_b:
        da = {} 
        da['id']= int(i)
        da["label"] = data1[int(i)][1]
        da[f"{modela_name}_answer"] = data1[int(i)][3]
        da[f"{modelb_name}_answer"] = data1[int(i)][4]
        writer.write(da)
with jsonlines.open(path3+"/draw.jsonl",'w') as writer:
    da = {}
    da[f"{modela_name}_win_number"] = len(winner_a)
    da[f"{modelb_name}_win_number"] = len(winner_b)
    da["draw_number"] = len(winner_draw)
    writer.write(da)
    for i in winner_draw:
        da = {} 
        da['id']= int(i)
        da["label"] = data1[int(i)][1]
        da[f"{modela_name}_answer"] = data1[int(i)][3]
        da[f"{modelb_name}_answer"] = data1[int(i)][4]
        writer.write(da)