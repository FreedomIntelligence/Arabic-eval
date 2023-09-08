import pandas as pd
import json
import jsonlines
#传进来的json应该有"id",
path1 = "chatgpt.jsonl"#测评的第一个文件
path2 = "llamaace-v2.jsonl"#测评的第二个文件
data = pd.read_json(path1,lines = True)
data1 = pd.read_json(path2,lines = True)
for i in range(len(data)):
    with jsonlines.open('combine1.jsonl', mode='w') as writer:
        for i in range(len(data)):
            da = {}
            da["id"]=int(data["id"][i])
            da["question"] = data["query"][i]
            da["answer1"] = data["output"][i]
            da["answer2"] = data1["output"][i]
            writer.write(da)


