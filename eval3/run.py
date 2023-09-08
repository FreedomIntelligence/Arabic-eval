import json
import jsonlines
import pandas as pd
path1 = "ACEGPT-v5.2.jsonl"
path2 = "llama2-7B-chat.jsonl"
data1 = pd.read_json(path1,lines = True)
data2 = pd.read_json(path2,lines = True)
with jsonlines.open("combine.jsonl",'w') as writer:
    for i in range(len(data1)):
        da = {}
        da['id'] = int(i+1)
        da['category'] = data1['label'][i]
        da['question'] = data1['query'][i]
        da['answer1'] = data1['output'][i]
        da['answer2'] = data2['output'][i]
        writer.write(da)
