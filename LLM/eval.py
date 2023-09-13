from accelerate import Accelerator
import argparse
from utils import Agent, GPTAgent
from benchmark_eval import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--generation_type', type=str, default='greedy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_name', type=str)
    parser.add_argument('--setting', type=str)
    parser.add_argument('--n_shot', type=int, default=None)
    args = parser.parse_args()

    accelerator = Accelerator()
    if args.model_id == 'gpt-3.5-turbo':
        agent = GPTAgent.from_model_id(args.model_id, generation_type=args.generation_type, accelerator=accelerator)
    else:
        agent = Agent.from_model_id(args.model_id, generation_type=args.generation_type, accelerator=accelerator)
    
    eval_class = benchmark2class[args.benchmark_name]
    Eval = eval_class(agent=agent,
                      batch_size=args.batch_size,
                      setting=args.setting,
                      n_shot=args.n_shot,
                      accelerator=accelerator)
    Eval.evaluate()


