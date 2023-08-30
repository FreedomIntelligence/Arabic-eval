# ALUE
#### train:  
      90% of all train datasets,and a dataset named `final.jsonl` is the result of combining all the training data in train folder  

#### dev:  
      10% of all train datasets which are made by ourself to verify our results.

#### original_dev:   
      Original dev datasets from ALUE.org  

#### test:  
      Test datasets from ALUE.org  

#### submit_run.py:  
      This file is used to process the generate results into suitable format to submit to ALUE.org.  
      To run thsi file,you have to:  
      1.Put your generate data into data_generate.  
      2.Rename the file into a right name:"{task}_test.jsonl"  
      3.There is a list about right names:["SEC_test.jsonl","MQ2Q_test.jsonl","FID_test.jsonl","MDD_test.jsonl","OOLD_test.jsonl","OHSD_test.jsonl","SVREG_test.jsonl","XNLI_test.jsonl","DIAG_test.jsonl"]  
      4.The right format files which are suitabel for ALUE.org will be saved in  ``predictions``  after runing the submit_sun.py


#### To generate data:  
If you want to generate data,you have to put the dateset which you want to test in bloom_eval/LLM-eval_pipeline-main/generate/data/ALUE,and rename it.  

For example,data's name is "MDD_test.jsonl",you have to run this code:  
```dos
python generate.py \
    --model_id ${model_id=bloom_ALUE_v2} \
    --generation_type ${generation_type=greedy} \
    --data_name ${data_name=ALUE/MDD_test} \
    --batch_size ${batch_size=4}
```
  
  The result will be save at "generate/output/ALUE/MDD_test/bloom_ALUE_v2.jsonl"
