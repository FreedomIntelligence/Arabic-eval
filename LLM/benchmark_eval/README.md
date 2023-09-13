# Benchmark Evaluation

Evaluation code for benchmarks with ground truth.


## Supporting Benchmarks

|  Benchmark  | Language |  Supporting  |
|-------------|----------|--------------|
|    MMCU     |    zh    |      ✅      |
|    GSM8K    |    en    |      ✅      |
|    MMLU     |    en    |      ✅      |
|    BBH      |    en    |      ❌      |



## Usage

(There are the corresponding scripts named by `eval_{benchmark}.py`)

You should config
- `model_id`: specify the model
- `generate_type`: specify the decoding hyperparameter config, which is `beam` by default (see `LLM-eval-pipeline/gconfig`)
- `setting`: `zero_shot`, `zero_shot_cot`, or `few_shot`

Here, `model_id` and `generate_type` are shared in this project, while `setting` is specific in this part.


### Outputs
Evaluation results, including model_output_file and metric_file, are saved to `results/$setting/$benchmark_name/$model_id/`

```
zero_shot/
    GSM8K/
        phoenix-inst-chat-7b/
            metrics.json
            output.jsonl
        chimera-inst-chat-7b/
    MMCU/
...

few_shot/
    GSM8K/
    MMCU/
    ...
    
zero_shot_cot/
    GSM8K/
    MMCU/
    ...
```

where, 
- `metrics.json`: The metrics, which are different across benchmarks. Here is an example from MMCU,
    ```json
    {
        "教育": {
            "化学": {
                "Accuracy": 0.20833333333333334
            },
            "历史": {
                "Accuracy": 0.35648148148148145
            },
            ...
            "overall": {
                "Accuracy": 0.29604710131405826
            }
        },
        "法律": {
            "Accuracy": 0.2914775592536561
        },
        ...
    }
    ```

- `output.jsonl` (names are different across benchmarks): Files storing both input (and the ground truth) and model outputs (and the corresponding decoded answers). Here is an example from MMCU,
    ```json
    {
    "query_id": 0,
    "query": "我国法定的甲类传染病包括：（）\nA. 鼠疫、霍乱\nB. 鼠疫、霍乱、天花\nC. 鼠疫、霍乱、爱滋病\nD. 天花、霍乱",
    "answer": "A",
    "prompted_query": "请阅读以下选择题并给出正确选项，不要解释原因。请只给出答案的序号。\n我国法定的甲类传染病包括：（）\nA. 鼠疫、霍乱\nB. 鼠疫、霍乱、天花\nC. 鼠疫、霍乱、爱滋病\nD. 天花、霍乱",
    "response": "正确选项是 A. 鼠疫、霍乱。",
    "response_answer": "A",
    },
    ```





## Add new benchmarks
To add new benchmarks, you should 
1. preprocess and save the data
2. config the benchmark in code

### Preprocess data
Put the benchmark under `benchmarks/`. First save the original files in `$benchmark_name/origin/`. Standardize the data and save to `$benchmark_name/converted/`. 

Standardized data is comprised of `test.jsonl` and `few_shot_prompt`:
1. `test.jsonl`: the format is
    ```json
    {"query_id":0,"query":"","answer":""}
    ```
    - `query`: task input, without any other prompt (e.g. system prompt)
    - `answer`: task output, i.e. ground truth

    Here is an example from MMCU,
    ```json
    {
    "query_id": 0,
    "query": "我国法定的甲类传染病包括：（）\nA. 鼠疫、霍乱\nB. 鼠疫、霍乱、天花\nC. 鼠疫、霍乱、爱滋病\nD. 天花、霍乱",
    "answer": "A",
    },
    ```

2. `few_shot_prompt`: demonstrations + placeholder (`{test_question}`). For example,
    ```
    关于呼吸机相关性肺炎列描述正确的是
    A. 呼吸机辅助48小时
    B. 呼吸机辅助72小时
    C. 呼吸机辅助96小时
    D. 呼吸机辅助24小时
    正确答案的序号是： A

    关于危重患者肠外营养支持的护理措施下列叙述哪项不正确
    A. 妥善固定输注导管，避免扯脱
    B. 感染
    C. 排斥反应
    D. 胰瘘
    正确答案的序号是： D

    {test_question}
    正确答案的序号是：
    ```

Your benchmark may be made up of several subtasks, you should set each subtask as a folder. Overall, the directory is like
```
GSM8K/
    origin/
    converted/
        test.jsonl
        few_shot_prompt

MMCU/
    origin/
    converted/
        心理/
            test.jsonl
            few_shot_prompt
        法律/
            test.jsonl
            few_shot_prompt
        ...
...
```




### Config in code
The corresponding two files under the `beval_utils/`:
- `data_utils.py`: class for benchmarks - config, task prompts, dataset loading, and data processing
- `evaluation.py`: class for evaluation - generate, evaluate, metric, and save


You should set a class for the new benchmark in the two python files respectively. See the corresponding instructions in the annotation in `BenchmarkBase` and `EvaluationBase` respecitvely. You can also see specific implementation examples in these two files.



