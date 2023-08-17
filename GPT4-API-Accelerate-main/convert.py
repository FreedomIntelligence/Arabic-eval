import json

def convert_json_to_jsonl(json_file, output_file):
    with open(json_file, 'r',encoding='utf-8') as file:
        data = json.load(file)

    with open(output_file, 'w',encoding='utf-8') as file:
        data=data[:200]
        for item in data:
            conversations = item['conversations']
            instruction = next(conv['value'] for conv in conversations if conv['from'] == 'human')
            output = next(conv['value'] for conv in conversations if conv['from'] == 'gpt')

            dict_item = {
                'id': item['id'],
                'instruction': instruction,
                'input': '',
                'output': output
            }
            file.write(json.dumps(dict_item,ensure_ascii=False) + '\n')

# 示例使用方法
json_file = 'inst_evol_inst_cn.json'
output_file = 'input.jsonl'
convert_json_to_jsonl(json_file, output_file)


