# How to run ?
## Deploy to the server for the native speakers to evaluate
1. Edit `run_review.sh`, you shuold fill in a correct file address where you save the combine file in `file_path`, and right model names in `modela_name` and `modelb_name`.  

2. Make sure that the result is not totally finished. This program can not overwrite the original result when you run a new task, if you do have this requirement,please delete the result file.  

3. Run this instruction
   ``` dos
   bash run_review.sh
   ```
4. Edit `run_result.sh`, you shuold fill in a correct file address as same as `file_path` in `path1`; fill in a correct file address where you save the result file in `path2`;and right model names as same as namse in `run_review` in `modela_name` and `modelb_name`.
5. Run this instruction
   ``` dos
   bash run_result.sh
   ```
