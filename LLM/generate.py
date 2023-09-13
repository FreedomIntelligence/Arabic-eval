from accelerate import Accelerator
import argparse
from utils import Agent
from generation import Generation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--generation_type', type=str, default='greedy')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--piece_size', type=int, default=-1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device_placement', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()
    agent = Agent.from_model_id(args.model_id, generation_type=args.generation_type, accelerator=accelerator)
    Gen = Generation(agent=agent,
                     data_name=args.data_name,
                     batch_size=args.batch_size,
                     piece_size=args.piece_size,
                     resume=args.resume,
                     device_placement=args.device_placement,
                     accelerator=accelerator)
    Gen.generate()

