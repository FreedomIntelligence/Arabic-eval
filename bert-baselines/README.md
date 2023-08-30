# Prerequisites
Install python dependencies:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Code was tested on `Python3.6` and `Ubuntu 18.04`

# How to run
There are 4 arguments you have to prepare before run `run_alue.py`.    

## task_name
This argument is which task you want to test.  You can check the following to decied which name is needed:
```python
alue_output_modes = {
    "mq2q": "classification",
    "mdd": "classification",
    "fid": "classification",
    "svreg": "regression",
    "sec": "multilabel",
    "oold": "classification",
    "ohsd": "classification",
    "xnli": "classification",
    "diag": "classification"
}
```
## base_dir
This argument attach to the base label dataset which you will compare with your generate results.  

#### 1.Task has original dev dataset or test dataset with labels
 Just set this argument to `dev` or `test` depend on your will.  

#### 2.Dataset is not original
Then you have to set this augment to the path where you save the base label dataset.    

## generate_dir
Set this augment to the path where you save the generate dataset.  
## generate_dir
Set this augment to decide where you will save the metric.
## There is an integrity example:  
```dos
python run_alue.py \
--task_name diag \
--base_dir dev \
--generate_dir v2/diag_dev.jsonl \
--output_dir results/diag_v2
```
    
