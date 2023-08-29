import json
import pandas as pd
import os
import jsonlines
dirs = os.listdir("./")
print(dirs)
with jsonlines.open("./final.jsonl",'w')as writer:
    for dir in dirs:
        if (dir=="merge.py"):
            continue
        else:
            start = 1
            print(dir)
            data = pd.read_json(dir,lines = True)
            dir_name = dir.split('.')[0]
            for i in range(len(data)):
                da = {}
                id = dir_name+f"-{start}"
                da["id"] =str(id)
                start+=1
                da["query"] = data["query"][i]
                da["answer"]=str(data["answer"][i])
                writer.write(da)