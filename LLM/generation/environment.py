import numpy as np
import json
import os
from typing import Dict, Tuple, Sequence, List, Union



class Environment:
    """
    Load dataset (File: generation/data/$data_name/...)
    Process dataset

    Args:
        data_name (`str`)
            Name of the data, which will be used to find the data file in storage
    """

    def __init__(self, 
                 data_name: str):
        self.data_name = data_name
        self.data_file = os.path.join('generation/data', f'{data_name}.jsonl')

        self.dataset = self.get_dataset(self.data_file)

    @staticmethod
    def _load_jsonl(file: str) -> List[dict]:
        assert isinstance(file, str) and (file.endswith('.jsonl') or file.endswith('.jl'))
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(line) for line in lines]
        return data_list

    def _preprocess_dataset(self, dataset: List[dict]) -> List[dict]:
        """Preprocessing"""
        return [
            {
                **example,
                'processed_query': example['query'].strip(),
             } 
            for example in dataset
        ]

    def get_dataset(self, file: str) -> List[dict]:
        """Pipeline"""
        dataset = self._load_jsonl(file)
        dataset = self._preprocess_dataset(dataset)
        return dataset




