from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate import Accelerator
import argparse
import torch
from utils import Agent
from utils.configs import load_gconfig
from generation import Generation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, default="{question}")
    parser.add_argument('--fp32', action='store_true')
    
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--piece_size', type=int, default=-1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device_placement', action='store_true')

    parser.add_argument('--generation_type', type=str, default=None)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--min_length', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--min_new_tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    args = parser.parse_args()

    if args.fp32:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token="<unk>"))
    
    if args.generation_type is not None:
        gconfig = load_gconfig(args.generation_type)
    else:
        gconfig = GenerationConfig(
            do_sample=args.do_sample,
            max_length=args.max_length,
            min_length=args.min_length,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    accelerator = Accelerator()
    agent = Agent(model_id=args.model_id, model=model, tokenizer=tokenizer, stage=args.stage, system_prompt=args.system_prompt, gconfig=gconfig, accelerator=accelerator)
    Gen = Generation(agent=agent,
                     data_name=args.data_name,
                     batch_size=args.batch_size,
                     piece_size=args.piece_size,
                     resume=args.resume,
                     device_placement=args.device_placement,
                     accelerator=accelerator)
    Gen.generate()








