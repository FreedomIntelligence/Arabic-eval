import json

# 读取jsonl文件，使用utf-8编码
with open('../final/final.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = []
ids = []
# 处理每一行
for line in lines:
    item = json.loads(line)
    conversation = [
        {
            "from": "human",
            "value": item["query"]
        },
        {
            "from": "gpt",
            "value": item["answer"]
        }
    ]
    data.append({
        "conversations": conversation,
        "id": item["id"]
    })
    ids.append(item['id'])

# 写入json文件，使用utf-8编
with open('ALUE-09-06.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)