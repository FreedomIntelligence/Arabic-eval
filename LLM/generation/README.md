# Generation



## Usage
You should config
- `model_id`: specify the model
- `generate_type`: specify the decoding hyperparameter config, which is `beam` by default (see `LLM-eval-pipeline/gconfig`)
- `data_name`: specify the data name you want to generate responses for, which is put under the `data/`

Here, `model_id` and `generate_type` are shared in this project, while `data_name` is specific in the `generation` part.

```bash
python generate.py
```

### Prepare your data
Be sure that dataset have been put in `data/` and is named as `$data_name.jsonl`. The file format should be
```json
{"id": 1, "label": "", "query": ""}
```

You can see some examples in the project by yourself

### Outputs
The output file will be saved to `outputs/$data_name/$model_id.jsonl`
```json
{"id": 1, "label": "", "query": "", "output": ""}
```



## Usage for alpaca_eval
You should config
- `model_id`: specify the model

```bash
python alpaca_eval.py
```

The output file will be saved to `outputs/alpaca_eval/$model_id.json`
