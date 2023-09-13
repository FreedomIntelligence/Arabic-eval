import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from typing import Dict, Tuple, Sequence, List, Union
import json
import re
from tqdm import tqdm
import gc
from accelerate import Accelerator
import math
import shutil
from utils import Agent
from generation.environment import *



class Generation:
    """
    The generation pipeline:
        load preprocessed dataset (from environment classes)
        model generation
        save output files (TO: generation/outputs/$data_name/$model_id.jsonl)
    """
    # output: model_results_dir - generation/outputs/$data_name/$model_id.jsonl
    results_dir = 'generation/outputs'
    def __init__(self, 
                 agent: Agent, 
                 data_name: str,
                 batch_size: int,
                 piece_size: int = -1,
                 resume: bool = False,
                 device_placement: bool = True,
                 accelerator: Accelerator = None):
        self.agent = agent
        self.data_name = data_name
        self.batch_size = batch_size
        self.piece_size = piece_size
        self.resume = resume
        self.device_placement = device_placement
        self.accelerator = accelerator

        self.model_id = self.agent.model_id
        self.environment = Environment(data_name)
        self.dataset = self.environment.dataset
        self.collate_fn = self.agent.collate_fn
        
        self.data_results_dir = os.path.join(self.results_dir, self.data_name)
        self.output_file = os.path.join(self.data_results_dir, f'{self.model_id}.jsonl')
        # self._create_folder(self.data_results_dir)

        assert self.model_id != 'gpt-3.5-turbo', "No support"

    @staticmethod
    def _create_folder(path: str):
        """Create a folder for the path if there isn't"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def rm_file(file: str):
        if os.path.exists(file):
            os.remove(file)
    
    @staticmethod
    def rm_folder(folder: str):
        if os.path.exists(folder):
            shutil.rmtree(folder)

    @staticmethod
    def _load_jsonl(path: str):
        return [json.loads(line) for line in open(path).read().strip().split('\n')]

    def _generate_responses(self, dataset: List[dict]) -> List[str]:
        """Generate responses"""
        assert isinstance(dataset, list) and (len(dataset) == 0 or (len(dataset) != 0 and isinstance(dataset[0], dict)))

        response_list = self.agent.generate_from_dataset(
            query_list=[data['processed_query'] for data in dataset], 
            batch_size=self.batch_size, 
            device_placement=self.device_placement,
        )
        return response_list
        
    def _generate_responses_pieces(self, dataset: List[dict], piece_size: int = 1000) -> List[str]:
        """Generate responses"""
        assert isinstance(dataset, list) and ((len(dataset) != 0 and isinstance(dataset[0], dict)) or len(dataset) == 0)

        n_all = len(dataset)
        n_iter = math.ceil(n_all / piece_size)
        itr = tqdm(range(n_iter)) if self.accelerator.is_main_process else range(n_iter)
        
        all_response_list = []
        for i in itr:
            cur_dataset = dataset[i * piece_size: min((i+1) * piece_size, n_all)]
            response_list = self.agent.generate_from_dataset(
                query_list=[data['processed_query'] for data in cur_dataset], 
                batch_size=self.batch_size, 
                description=f'{i+1}-th/{n_iter} piece',
                device_placement=self.device_placement,
            )
            if self.accelerator.is_main_process:
                result_list = self._post_process(response_list)
                result_list = self._formalize_for_saving(cur_dataset, result_list)
                self._save_output(result_list, self.output_file, save_type='a+')
                
            all_response_list.extend(response_list)
            gc.collect(); torch.cuda.empty_cache()
                
        return all_response_list

    def _formalize_for_saving(self, data_list: List[dict], responses: List[str]) -> List[dict]:
        """Pack the response of each sample with the corresponding query"""
        assert isinstance(data_list, list) and (len(data_list) == 0 or (len(data_list) != 0 and isinstance(data_list[0], dict)))
        assert isinstance(responses, list) and (len(responses) == 0 or (len(responses) != 0 and isinstance(responses[0], str)))
        assert len(data_list) == len(responses), f'data_list and responses must have the same length, but are {len(data_list)} and {len(responses)}'

        samples = []
        for data, response in zip(data_list, responses):
            sample = {
                **data,
                'output': response,
            }
            sample['query'] = sample.pop('processed_query')

            samples.append(sample)
        return samples

    def _post_process(self, text_list: List[str]):
        return [text.strip() for text in text_list]
        
    def _save_output(self, samples: List[dict], save_file: str, save_type: str='w'):
        """Save responses to `$outputs/$data_name/$model_id.jsonl`"""
        assert isinstance(save_file, str)
        assert isinstance(samples, list) and isinstance(samples[0], dict)

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, save_type, encoding='utf-8') as f:
            f.writelines([json.dumps(sample, ensure_ascii=False) + '\n'  for sample in samples])
        print(f'Save to {save_file}')

    def skip_done(self, dataset: List[dict]) -> List[str]:
        done_data_list = []
        if os.path.exists(self.output_file):
            done_data_list = self._load_jsonl(self.output_file)
            
        done_id_list = [x['id'] for x in done_data_list]
        remain_data_list = list(filter(lambda x: x['id'] not in done_id_list, dataset))
        return done_data_list, remain_data_list
    
    def generate(self):
        """Pipeline"""
        done = []
        dataset = self.dataset
        if self.resume:
            done, dataset = self.skip_done(dataset)
        else:
            if self.accelerator.is_main_process:
                self.rm_file(self.output_file)
            
        if self.piece_size != -1:
            response_list = self._generate_responses_pieces(dataset, piece_size=self.piece_size)
        else:
            response_list = self._generate_responses(dataset)
            
        response_list = self._post_process(response_list)
        result_list = self._formalize_for_saving(dataset, response_list)
        if self.accelerator.is_main_process:
            self._save_output(done + result_list, self.output_file)



