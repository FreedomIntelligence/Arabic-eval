import numpy as np
import json
import os
import pandas as pd
import glob
from typing import Dict, Tuple, Sequence, List, Union
from collections import defaultdict



class BenchmarkBase:
    """
    Loading dataset and few_shot_prompt
    Process dataset for zero_shot or few_shot setting

    Subclass should config:
        benchmark_name (`str`)
        benchmark_dir (`str`)
        subtasks (`List[str]`)

    Subclass should implement:
        _init_task_prompts(self):
            config task_prompt for each setting (and for each subtask)
    """

    subtasks = []
    benchmarks_dir = 'benchmark_eval/benchmarks'

    def __init__(
        self, 
        benchmark_name: str = ''
    ):
        self.benchmark_name = benchmark_name

        self.benchmark_dir = os.path.join(self.benchmarks_dir, benchmark_name)
        self._init_task_prompts()
        self._normalize_task_prompts()

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        raise NotImplementedError

    def _normalize_task_prompts(self):
        if isinstance(self.task_prompt_zero_shot, str):
            self.task_prompt_zero_shot = {self.benchmark_name: self.task_prompt_zero_shot}
        if isinstance(self.response_with_zero_shot, str):
            self.response_with_zero_shot = {k: self.response_with_zero_shot for k in self.task_prompt_zero_shot.keys()}
        if isinstance(self.task_prompt_few_shot, str):
            self.task_prompt_few_shot = {self.benchmark_name: self.task_prompt_few_shot}

    @staticmethod
    def _read_txt(file: str) -> List[dict]:
        assert isinstance(file, str)
        with open(file, 'r') as f:
            return f.read()

    @staticmethod
    def _load_jsonl(file: str) -> List[dict]:
        assert isinstance(file, str) and (file.endswith('.jsonl') or file.endswith('.jl'))
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(line) for line in lines]
        return data_list

    @staticmethod
    def _load_csv(file: str, header=None):
        assert isinstance(file, str) and file.endswith('.csv')
        return pd.read_csv(file, header=header)

    @staticmethod
    def _load_excel(file: str, sheet_name=None, header=None):
        assert isinstance(file, str) and file.endswith('.xlsx')
        return pd.read_excel(file, sheet_name=sheet_name, header=header)

    @staticmethod
    def _get_sheet_names(file: str):
        assert isinstance(file, str) and file.endswith('.xlsx')
        return pd.ExcelFile(file).sheet_names

    def _normalize_datasets_and_few_shot_prompts(
        self, 
        datasets: Union[List[dict], Dict[str, List[dict]]], 
        few_shot_prompts: Union[str, Dict[str, str]]
    ) -> Dict[str, List[dict]]:
        if isinstance(datasets, list):
            datasets = {self.benchmark_name: datasets}
        if isinstance(few_shot_prompts, str):
            few_shot_prompts = {self.benchmark_name: few_shot_prompts}

        for task_name, few_shot_prompt in few_shot_prompts.items():
            few_shot_prompts[task_name] = few_shot_prompt.replace('{', '{{').replace('}', '}}').replace('{{test_question}}', '{test_question}')
        return datasets, few_shot_prompts

    def _prompting_for_zero_shot(self, dataset: List[dict], task_prompt: str='', response_with: str='') -> List[dict]:
        """Prompt one dataset for zero-shot setting"""
        return [
            {
                **data,
                'prompted_query': task_prompt.format(input=data['query'].strip()),
                'response_with': response_with,
            }
            for data in dataset
        ]

    def _prompting_for_few_shot(self, dataset: List[dict], few_shot_prompt: str='', task_prompt: str='') -> List[dict]:
        """Prompt one dataset for few-shot setting"""
        return [
            {
                **data,
                'prompted_query': task_prompt.format(
                    input=few_shot_prompt.format(test_question=data['query'].strip()),
                ),
                'response_with': '',
            }
            for data in dataset
        ]
    
    def _prompting_datasets(self, 
            datasets: Dict[str, List[dict]],
            few_shot_prompts: Dict[str, str],
            setting: str,
            stage: int,
        ) -> Dict[str, List[dict]]:
        """Prompt all datasets, available for zero_shot and few_shot settings"""
        assert isinstance(datasets, dict) and isinstance(few_shot_prompts, dict)

        assert set(datasets.keys()) == set(few_shot_prompts.keys())
        if setting == 'few_shot':
            return {
                task_name:  self._prompting_for_few_shot(datasets[task_name], few_shot_prompt=few_shot_prompts[task_name], task_prompt=self.task_prompt_few_shot[task_name])
                for task_name in datasets.keys()
            }
        else:
            return {
                task_name:  self._prompting_for_zero_shot(datasets[task_name], task_prompt=self.task_prompt_zero_shot[task_name], response_with=self.response_with_zero_shot[task_name] if stage == 1 else '')
                for task_name in datasets.keys()
            }

    def get_datasets(
        self, 
        setting: str = 'few_shot',
        n_shot: int = None,
        stage: int = 1,
    ) -> Union[List[dict], Dict[str, List[dict]]]:
        all_datasets: Union[List[dict], Dict[str, List[dict]]] = self._prepare_data()
        few_shot_prompts: Union[str, Dict[str, str]] = self._prepare_few_shot_prompt(n_shot=n_shot)
        all_datasets, few_shot_prompts = self._normalize_datasets_and_few_shot_prompts(all_datasets, few_shot_prompts)
    
        return self._prompting_datasets(all_datasets, few_shot_prompts, setting, stage)



class MMCUBenchmark(BenchmarkBase):

    top_subjects = ['心理', '医疗', '教育', '法律']
    subtasks = ['医疗_临床医学', '医疗_传染病学', '医疗_儿科学', '医疗_免疫学', '医疗_医学三基', '医疗_医学影像学', '医疗_外科学', '医疗_寄生虫学', '医疗_护理学', '医疗_病理学', '医疗_皮肤性病学', '医疗_组织胚胎学', '医疗_药物分析学', '医疗_药理学', '医疗_解剖学', 
                '教育_化学', '教育_历史', '教育_地理', '教育_政治', '教育_数学', '教育_物理', '教育_生物', '教育_语文', 
                '法律',
                '心理']

    def __init__(self):
        super(MMCUBenchmark, self).__init__(benchmark_name='MMCU')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        task_prompt = '请阅读以下选择题并给出正确选项，不要解释原因。请只给出答案的序号。\n{input}'
        response_with = '正确答案的序号是：'
        self.task_prompt_zero_shot = {
            task: task_prompt for task in self.subtasks
        }
        self.response_with_zero_shot = response_with

        self.task_prompt_few_shot = {
            task: task_prompt for task in self.subtasks
        }

    def _prepare_data(self) -> Dict[str, List[dict]]:
        task_list = glob.glob(os.path.join(self.benchmark_dir, 'converted', '*'))
        dataset_dict = {}
        for task in task_list:
            data_list = self._load_jsonl(os.path.join(task, 'test.jsonl'))
            task_name = os.path.basename(task)
            dataset_dict[task_name] = data_list
        return dataset_dict

    def _prepare_few_shot_prompt(self, n_shot = 5) -> Dict[str, str]:
        if n_shot is None:
            n_shot = 5
        assert n_shot == 5, ('Only support 5-shot')

        task_list = glob.glob(os.path.join(self.benchmark_dir, 'converted', '*'))
        few_shot_prompt_dict = {}
        for task in task_list:
            few_Shot_prompt = self._read_txt(os.path.join(task, 'few_shot_prompt'))
            task_name = os.path.basename(task)
            few_shot_prompt_dict[task_name] = few_Shot_prompt

        return few_shot_prompt_dict


class MMLUBenchmark(BenchmarkBase):

    top_subjects = ['physics', 'chemistry', 'biology', 'computer science', 'math', 'engineering', 'history', 'philosophy', 'law', 'politics', 'culture', 'economics', 'geography', 'psychology', 'other', 'business', 'health']
    subtasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    categories = {'math': ['abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics'], 'health': ['anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'medical_genetics', 'nutrition', 'professional_medicine', 'virology'], 'physics': ['astronomy', 'college_physics', 'conceptual_physics', 'high_school_physics'], 'business': ['business_ethics', 'management', 'marketing'], 'biology': ['college_biology', 'high_school_biology'], 'chemistry': ['college_chemistry', 'high_school_chemistry'], 'computer science': ['college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning'], 'economics': ['econometrics', 'high_school_macroeconomics', 'high_school_microeconomics'], 'engineering': ['electrical_engineering'], 'philosophy': ['formal_logic', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'world_religions'], 'other': ['global_facts', 'miscellaneous', 'professional_accounting'], 'history': ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'prehistory'], 'geography': ['high_school_geography'], 'politics': ['high_school_government_and_politics', 'public_relations', 'security_studies', 'us_foreign_policy'], 'psychology': ['high_school_psychology', 'professional_psychology'], 'culture': ['human_sexuality', 'sociology'], 'law': ['international_law', 'jurisprudence', 'professional_law']}
    top_categories = {
        "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other (business, health, misc.)": ["other", "business", "health"],
    }
    def __init__(self):
        super(MMLUBenchmark, self).__init__(benchmark_name='MMLU')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        self.task_prompt_zero_shot = {
            task: 'The following are multiple choice questions about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        }
        self.response_with_zero_shot = '\nAnswer:'

        self.task_prompt_few_shot = {
            task: 'The following are multiple choice questions (with answers) about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        }

    def _prepare_data(self) -> Dict[str, List[dict]]:
        data_files = glob.glob(os.path.join(self.benchmark_dir, 'test', '*.csv'))

        def normalize(x):
            return str(x).strip()

        dataset_dict = {}
        for file in data_files:
            task_name = os.path.basename(file).split('_test')[0]
            df = self._load_csv(file, header=None)

            processed_data_list = []
            for index, row in df.iterrows():
                processed_data_list.append({
                    'query_id': index,
                    'query': f"{normalize(row[0])}\nA. {normalize(row[1])}\nB. {normalize(row[2])}\nC. {normalize(row[3])}\nD. {normalize(row[4])}",
                    'answer': f"{normalize(row[5])}",
                })
            dataset_dict[task_name] = processed_data_list

        return dataset_dict

    def _prepare_few_shot_prompt(self, n_shot = 5) -> Dict[str, str]:
        if n_shot is None:
            n_shot = 5
        assert n_shot <= 5, ('At most 5-shot')

        data_files = glob.glob(os.path.join(self.benchmark_dir, 'dev', '*.csv'))

        def normalize(x):
            return str(x).strip()

        few_shot_prompt_dict = {}
        for file in data_files:
            task_name = os.path.basename(file).split('_dev')[0]
            df = self._load_csv(file, header=None)

            few_shot_prompt = ''
            for index, row in df.iterrows():
                if index == n_shot:
                    break
                few_shot_prompt += (
                    f"{normalize(row[0])}\nA. {normalize(row[1])}\nB. {normalize(row[2])}\nC. {normalize(row[3])}\nD. {normalize(row[4])}"
                    + f"\nAnswer: {normalize(row[5])}\n\n"
                )
            few_shot_prompt += "{test_question}\nAnswer:"
            few_shot_prompt_dict[task_name] = few_shot_prompt

        return few_shot_prompt_dict


class GSM8KBenchmark(BenchmarkBase):

    def __init__(self):
        super(GSM8KBenchmark, self).__init__(benchmark_name='GSM8K')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        self.task_prompt_zero_shot = 'Question: {input}'
        self.response_with_zero_shot = '\nThe answer (arabic numerals) is'

        self.task_prompt_few_shot = '{input}'
        
    def _prepare_data(self) -> List[dict]:
        data_list = self._load_jsonl(os.path.join(self.benchmark_dir, 'test.jsonl'))

        processed_data_list = []
        for i, data in enumerate(data_list):
            processed_data_list.append({
                'query_id': i,
                'query': data['question'],
                'answer': data['answer'].split('#### ')[-1].strip(),
            })
        return processed_data_list
    
    def _prepare_few_shot_prompt(self, n_shot = 8) -> str:
        if n_shot is None:
            n_shot = 8
        assert n_shot == 8, ('Only support 8-shot')

        prompt_file = os.path.join(self.benchmark_dir, 'few_shot_prompt')
        with open(prompt_file) as f:
            return f.read()


class CEvalBenchmark(BenchmarkBase):
    # TODO
    # benchmark_dir = os.path.join('benchmark_eval/benchmarks', 'CEval', 'converted')

    top_subjects = ['STEM', 'Social Science', 'Humanity', 'Other']
    subtasks = ['business_administration', 'art_studies', 'sports_science', 'middle_school_geography', 'probability_and_statistics', 'plant_protection', 'chinese_language_and_literature', 'environmental_impact_assessment_engineer', 'discrete_mathematics', 'middle_school_politics', 'advanced_mathematics', 'college_economics', 'tax_accountant', 'basic_medicine', 'operating_system', 'computer_network', 'metrology_engineer', 'law', 'education_science', 'urban_and_rural_planner', 'college_programming', 'legal_professional', 'logic', 'high_school_geography', 'clinical_medicine', 'ideological_and_moral_cultivation', 'high_school_mathematics', 'high_school_history', 'modern_chinese_history', 'middle_school_mathematics', 'teacher_qualification', 'accountant', 'middle_school_history', 'mao_zedong_thought', 'middle_school_chemistry', 'high_school_politics', 'college_chemistry', 'marxism', 'computer_architecture', 'middle_school_biology', 'veterinary_medicine', 'high_school_chemistry', 'high_school_biology', 'professional_tour_guide', 'physician', 'electrical_engineer', 'fire_engineer', 'college_physics', 'high_school_physics', 'high_school_chinese', 'civil_servant', 'middle_school_physics']
    top_categories = {
        "STEM": ["electrical_engineer", "metrology_engineer", "college_programming", "computer_architecture", "operating_system", "computer_network", "discrete_mathematics", "probability_and_statistics", "advanced_mathematics", "college_chemistry", "college_physics", "veterinary_medicine", "high_school_biology", "high_school_chemistry", "high_school_physics", "high_school_mathematics", "middle_school_chemistry", "middle_school_physics", "middle_school_biology", "middle_school_mathematics"],
        "Humanities": ["environmental_impact_assessment_engineer", "urban_and_rural_planner", "fire_engineer", "physician", "tax_accountant", "accountant", "civil_servant", "clinical_medicine", "basic_medicine", "plant_protection", "sports_science"],
        "Social Sciences": ["teacher_qualification", "business_administration", "mao_zedong_thought", "marxism", "college_economics", "education_science", "high_school_geography", "high_school_politics", "middle_school_geography", "middle_school_politics"],
        "Other": ["professional_tour_guide", "legal_professional", "art_studies", "chinese_language_and_literature", "law", "logic", "ideological_and_moral_cultivation", "modern_chinese_history", "high_school_history", "high_school_chinese", "middle_school_history"],
    }
    def __init__(self):
        super(CEvalBenchmark, self).__init__(benchmark_name='CEval')

    def _init_task_prompts(self):
        self.task_prompt_zero_shot = self.task_prompt_few_shot = {
            task: '以下是中国关于%s考试的单项选择题，请选出其中的正确答案。\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        }        




class MMLUArbicBenchmark(BenchmarkBase):

    top_subjects = ['physics', 'chemistry', 'biology', 'computer science', 'math', 'engineering', 'history', 'philosophy', 'law', 'politics', 'culture', 'economics', 'geography', 'psychology', 'other', 'business', 'health']
    subtasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    subtasks_ar = ['جبر_تجريدي', 'تشريح', 'علم_الفلك', 'أخلاقيات_الأعمال', 'المعرفة_السريرية', 'علم_الأحياء_الجامعي', 'كيمياء_جامعية', 'علوم_الحاسوب_الجامعية', 'رياضيات_جامعية', 'طب_جامعي', 'فيزياء_جامعية', 'أمان_الحاسوب', 'فيزياء_مفاهيمية', 'الاقتصاد_القياسي', 'هندسة_كهربائية', 'رياضيات_ابتدائية', 'منطق_رسمي', 'حقائق_عالمية', 'علم_الأحياء_الثانوي', 'كيمياء_ثانوية', 'علوم_الحاسوب_الثانوية', 'تاريخ_أوروبا_الثانوي', 'جغرافية_ثانوية', 'الحكومة_والسياسة_الثانوية', 'اقتصاد_كلي_ثانوي', 'رياضيات_ثانوية', 'اقتصاد_جزئي_ثانوي', 'فيزياء_ثانوية', 'علم_النفس_الثانوي', 'إحصاء_ثانوي', 'تاريخ_الولايات_المتحدة_الثانوي', 'تاريخ_العالم_الثانوي', 'شيخوخة_الإنسان', 'جنسانية_بشرية', 'قانون_دولي', 'فقه', 'أخطاء_منطقية', 'تعلم_الآلة', 'إدارة', 'تسويق', 'جينات_طبية', 'متفرقات', 'نزاعات_أخلاقية', 'سيناريوهات_أخلاقية', 'تغذية', 'فلسفة', 'ما_قبل_التاريخ', 'محاسبة_مهنية', 'قانون_مهني', 'طب_مهني', 'علم_النفس_المهني', 'علاقات_عامة', 'دراسات_الأمان', 'علم_الاجتماع', 'سياسة_خارجية_أمريكية', 'علم_الفيروسات', 'أديان_العالم']
    categories = {'math': ['abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics'], 'health': ['anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'medical_genetics', 'nutrition', 'professional_medicine', 'virology'], 'physics': ['astronomy', 'college_physics', 'conceptual_physics', 'high_school_physics'], 'business': ['business_ethics', 'management', 'marketing'], 'biology': ['college_biology', 'high_school_biology'], 'chemistry': ['college_chemistry', 'high_school_chemistry'], 'computer science': ['college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning'], 'economics': ['econometrics', 'high_school_macroeconomics', 'high_school_microeconomics'], 'engineering': ['electrical_engineering'], 'philosophy': ['formal_logic', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'world_religions'], 'other': ['global_facts', 'miscellaneous', 'professional_accounting'], 'history': ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'prehistory'], 'geography': ['high_school_geography'], 'politics': ['high_school_government_and_politics', 'public_relations', 'security_studies', 'us_foreign_policy'], 'psychology': ['high_school_psychology', 'professional_psychology'], 'culture': ['human_sexuality', 'sociology'], 'law': ['international_law', 'jurisprudence', 'professional_law']}
    top_categories = {
        "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other (business, health, misc.)": ["other", "business", "health"],
    }
    def __init__(self):
        super(MMLUArbicBenchmark, self).__init__(benchmark_name='MMLUArabic')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        # self.task_prompt_zero_shot = self.task_prompt_few_shot = {
        #     task: 'The following are multiple choice questions (with answers) about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        # }
        # self.task_prompt_zero_shot = {
        #     task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        # }

        self.task_prompt_zero_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}\nمن فضلك اختر إجابة واحدة من بين 'A، B، C، D' دون شرح." % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }

        
        self.response_with_zero_shot = '\nإجابة:'

        self.task_prompt_few_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد (مع الإجابات) حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }

    def _prepare_data(self) -> Dict[str, List[dict]]:
        data_files = glob.glob(os.path.join(self.benchmark_dir, 'test', '*.csv'))

        def normalize(x):
            return str(x).strip()

        dataset_dict = {}
        for file in data_files:
            task_name = os.path.basename(file).split('_test')[0]
            df = self._load_csv(file, header=None)

            processed_data_list = []
            for index, row in df.iterrows():
                processed_data_list.append({
                    'query_id': index,
                    'query': f"سؤال: {normalize(row[0])}\nA. "+ f"{normalize(row[1])}"+"\nB. "+ f"{normalize(row[2])}" +"\nC. " +f"{normalize(row[3])}" +"\nD. " +f"{normalize(row[4])}",
                    'answer': f"{normalize(row[5])}",
                })
            dataset_dict[task_name] = processed_data_list

        return dataset_dict

    def _prepare_few_shot_prompt(self, n_shot = 5) -> Dict[str, str]:
        if n_shot is None:
            n_shot = 5
        assert n_shot <= 5, ('At most 5-shot')

        data_files = glob.glob(os.path.join(self.benchmark_dir, 'dev', '*.csv'))

        def normalize(x):
            return str(x).strip()

        few_shot_prompt_dict = {}
        for file in data_files:
            task_name = os.path.basename(file).split('_dev')[0]
            df = self._load_csv(file, header=None)

            few_shot_prompt = ''
            for index, row in df.iterrows():
                if index == n_shot:
                    break
                few_shot_prompt += (
                    f"سؤال: {normalize(row[0])}\nA. "+ f"{normalize(row[1])}"+"\nB. "+ f"{normalize(row[2])}" +"\nC. " +f"{normalize(row[3])}" +"\nD. " +f"{normalize(row[4])}"
                    + "\n" +"إجابة: "+f"{normalize(row[5])}\n\n"
                )
            few_shot_prompt += "{test_question}\nإجابة:" 
            few_shot_prompt_dict[task_name] = few_shot_prompt

        return few_shot_prompt_dict


class EXAMSBenchmark(BenchmarkBase):

    subtasks = ['Islamic Studies', 'Science', 'Social', 'Biology', 'Physics']
    subtasks_ar = ['الدراسات الإسلامية', 'العلوم', 'الاجتماعيات', 'علم الأحياء', 'علم الفيزياء']
    def __init__(self):
        super(EXAMSBenchmark, self).__init__(benchmark_name='EXAMS_Arabic')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        # self.task_prompt_zero_shot = self.task_prompt_few_shot = {
        #     task: 'The following are multiple choice questions (with answers) about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        # }
        # self.task_prompt_zero_shot = {
        #     task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        # }

        self.task_prompt_zero_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}\nمن فضلك اختر إجابة واحدة من بين 'A، B، C، D' دون شرح." % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }


        self.response_with_zero_shot = '\nإجابة:'

        self.task_prompt_few_shot = {
            task: "فيما يلي أسئلة الاختيار من متعدد (مع الإجابات) حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].split('_')) for i,task in enumerate(self.subtasks)
        }

    def _prepare_data(self) -> Dict[str, List[dict]]:
        data_file = os.path.join(self.benchmark_dir, 'exam_test.jsonl')

        def normalize(x):
            return str(x).strip()

        dataset_dict = defaultdict(lambda: []) 

        data_list = self._load_jsonl(data_file)
        for data in data_list:
            c = data['id'].split('-')[0]
            dataset_dict[c].append({
                'query_id': data['id'],
                'query': f"سؤال: {normalize(data['question'])}\nA. "+ f"{normalize(data['A'])}"+"\nB. "+ f"{normalize(data['B'])}" +"\nC. " +f"{normalize(data['C'])}" +"\nD. " +f"{normalize(data['D'])}",
                'answer': normalize(data['answer']),
            })
        
        return dataset_dict
    
    def _prepare_few_shot_prompt(self, n_shot = 5) -> Dict[str, str]:
        if n_shot is None:
            n_shot = 5
        assert n_shot <= 5, ('At most 5-shot')

        data_file = os.path.join(self.benchmark_dir, 'exam_dev.jsonl')

        def normalize(x):
            return str(x).strip()

        few_shot_prompt_dict = defaultdict(lambda: '')
        count_shots = defaultdict(lambda: 0)

        data_list = self._load_jsonl(data_file)
        for data in data_list:
            c = data['id'].split('-')[0]
            if count_shots[c] == n_shot:
                continue

            few_shot_prompt_dict[c] += (
                f"سؤال: {normalize(data['question'])}\nA. "+ f"{normalize(data['A'])}"+"\nB. "+ f"{normalize(data['B'])}" +"\nC. " +f"{normalize(data['C'])}" +"\nD. " +f"{normalize(data['D'])}"
                + "\n" +"إجابة: "+f"{normalize(data['answer'])}\n\n"
            )
            count_shots[c] += 1

        for c in self.subtasks:
            few_shot_prompt_dict[c] += "{test_question}\nإجابة:" 

        return few_shot_prompt_dict
    


class ArabicCultureBenchmark(BenchmarkBase):

    subtasks = ['Arabic Funeral', 'Sudan', 'Arabic Physics and Chemistry', 'Algeria', 'InfluenceFromAncientEgypt', 'Arabic Ceremony', 'Arabic Astronomy', 'Arabic Calligraphy', 'daily life', 'Saudi Arabia', 'Arabic Language Origin', 'Arabic Ornament', 'Islamic law system', 'Kuwait', 'InfluenceFromChina', 'Arabic Literature', 'computer and phone', 'Tunisia', 'Arabic Geography', 'Arabic Music', 'Arabic Medicine', 'Arabic Philosophy', 'Yemen', 'Jordan', 'Mesopotamia civilization', 'Islam Education', 'Arabic Wedding', 'InfluenceFromRome', 'Egypt modern', 'Ancient Egypt', 'Comoros', 'InfluenceFromGreece', 'Qatar', 'InfluenceFromPersia', 'Lebanon', 'Arabic Math', 'Arabic History', 'InfluenceFromIslam', 'Libya', 'Syria', 'Oman', 'Arabic Culture', 'Arabic Art', 'United Arab Emirates', 'Islam branches and schools', 'InfluenceFromByzantium', 'Arab Empire', 'Arabic Food', 'Mauritania', 'entertainment', 'communication', 'Palestine', 'Bahrain', 'Somalia', 'Iraq', 'Arabic Clothing', 'Arabic Architecture', 'Morocco',
                'talks', 'traditional festival', 'life', 'specialExpression', 'Islam', 'traveling', 'trade and bussness']
    subtasks_ar = ['جنازة عربية', 'السودان', 'الفيزياء والكيمياء ', 'الجزائر', 'التأثير من مصر القديمة', 'المراسم العربية', 'الفلك العربي', 'الخط العربي', 'الحياة اليومية', 'المملكة العربية السعودية', 'أصل اللغة العربية', 'الزخرفة العربية', 'نظام القانون الإسلامي', 'الكويت', 'التأثير من الصين', 'الأدب العربي', 'الكمبيوتر والهاتف', 'تونس', 'الجغرافيا العربية', 'الموسيقى العربية', 'الطب العربي', 'الفلسفة العربية', 'اليمن', 'الأردن', 'حضارة بلاد الرافدين', 'التعليم الإسلامي', 'الزفاف العربي', 'التأثير من روما', 'مصر الحديثة', 'مصر القديمة', 'جزر القمر', 'التأثير من اليونان', 'قطر', 'التأثير من بلاد فارس', 'لبنان', 'الرياضيات ', 'التاريخ العربي', 'التأثير من الإسلام', 'ليبيا', 'سوريا', 'عمان', 'الثقافة العربية', 'الفن العربي', 'الإمارات العربية المتحدة', 'فروع الإسلام والمذاهب', 'التأثير من البيزنطيين', 'الإمبراطورية العربية', 'الطعام العربي', 'موريتانيا', 'الترفيه', 'الاتصالات', 'فلسطين', 'البحرين', 'الصومال', 'العراق', 'الملابس العربية', 'العمارة العربية', 'المغرب', 'المحادثات', 'المهرجانات التقليدية', 'الحياة', 'التعبيرات الخاصة', 'الإسلام', 'السفر', 'التجارة والأعمال']
    difficulty_levels = {
        'easy': ['Arabic Funeral', 'Sudan', 'Arabic Physics and Chemistry', 'Algeria', 'InfluenceFromAncientEgypt', 'Arabic Ceremony', 'Arabic Astronomy', 'Arabic Calligraphy', 'daily life', 'Saudi Arabia', 'Arabic Language Origin', 'Arabic Ornament', 'Islamic law system', 'Kuwait', 'InfluenceFromChina', 'Arabic Literature', 'computer and phone', 'Tunisia', 'Arabic Geography', 'Arabic Music', 'Arabic Medicine', 'Arabic Philosophy', 'Yemen', 'Jordan', 'Mesopotamia civilization', 'Islam Education', 'Arabic Wedding', 'InfluenceFromRome', 'Egypt modern', 'Ancient Egypt', 'Comoros', 'InfluenceFromGreece', 'Qatar', 'InfluenceFromPersia', 'Lebanon', 'Arabic Math', 'Arabic History', 'InfluenceFromIslam', 'Libya', 'Syria', 'Oman', 'Arabic Culture', 'Arabic Art', 'United Arab Emirates', 'Islam branches and schools', 'InfluenceFromByzantium', 'Arab Empire', 'Arabic Food', 'Mauritania', 'entertainment', 'communication', 'Palestine', 'Bahrain', 'Somalia', 'Iraq', 'Arabic Clothing', 'Arabic Architecture', 'Morocco'],
        'middle': ['talks', 'traditional festival', 'life', 'specialExpression', 'Islam', 'traveling', 'trade and bussness']
    }
    task_types = {
        'TF': difficulty_levels['easy'],
        'choices': difficulty_levels['middle']
    }

    def __init__(self):
        super(ArabicCultureBenchmark, self).__init__(benchmark_name='ArabicCulture')

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot and self.task_prompt_few_shot
        """
        # self.task_prompt_zero_shot = self.task_prompt_few_shot = {
        #     task: 'The following are multiple choice questions (with answers) about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        # }

        # self.task_prompt_zero_shot = {
        #     task: (
        #         "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input} " % ' '.join(self.subtasks_ar[i].lower()) if task in self.task_types['choices']
        #         else "فيما يلي أسئلة صحيحة أو خاطئة حول %s\n\n{input} " % ' '.join(self.subtasks_ar[i].lower()) # TODO
        #     )
        #     for i, task in enumerate(self.subtasks)
        # }
        
        self.task_prompt_zero_shot = {
            task: (
                "فيما يلي أسئلة الاختيار من متعدد حول %s\n\n{input}\n من فضلك اختر إجابة واحدة من بين 'A، B، C، D' دون شرح." % ' '.join(self.subtasks_ar[i].lower()) if task in self.task_types['choices']
                else "فيما يلي أسئلة صحيحة أو خاطئة حول %s\n\n{input}\n الرجاء إخراج 'صح' أو 'خطأ' دون شرح." % ' '.join(self.subtasks_ar[i].lower()) # TODO
            )
            for i, task in enumerate(self.subtasks)
        }
        
        self.response_with_zero_shot = '\nإجابة:'

        self.task_prompt_few_shot = {
            task: (
                "فيما يلي أسئلة الاختيار من متعدد (مع الإجابات) حول %s\n\n{input} " % ' '.join(self.subtasks_ar[i].lower()) if task in self.task_types['choices']
                else "فيما يلي أسئلة صحيحة أو خاطئة (مع الإجابات) حول %s\n\n{input}" % ' '.join(self.subtasks_ar[i].lower()) # TODO
            )
            for i, task in enumerate(self.subtasks)
        }

    def _prepare_data(self) -> Dict[str, List[dict]]:
        easy_file = os.path.join(self.benchmark_dir, 'difficulty-easy-TF_test.jsonl')
        middle_file = os.path.join(self.benchmark_dir, 'difficulty-mid-choice_test.jsonl')

        def normalize(x):
            return str(x).strip()

        dataset_dict = defaultdict(lambda: []) 

        easy_data_list = self._load_jsonl(easy_file)
        for data in easy_data_list:
            c = data['id'].split('-')[0]
            dataset_dict[c].append({
                'query_id': data['id'],
                'query': f"سؤال: {normalize(data['question'])}",
                'answer': normalize(data['answer']),
            })
        
        middle_data_list = self._load_jsonl(middle_file)
        for data in middle_data_list:
            c = data['id'].split('-')[0]
            dataset_dict[c].append({
                'query_id': data['id'],
                'query': f"سؤال: {normalize(data['question'])}\nA. "+ f"{normalize(data['A'])}"+"\nB. "+ f"{normalize(data['B'])}" +"\nC. " +f"{normalize(data['C'])}" +"\nD. " +f"{normalize(data['D'])}",
                'answer': normalize(data['answer']),
            })

        return dataset_dict
    
    def _prepare_few_shot_prompt(self, n_shot = 5) -> Dict[str, str]:
        if n_shot is None:
            n_shot = 5
        assert n_shot <= 5, ('At most 5-shot')

        easy_file = os.path.join(self.benchmark_dir, 'difficulty-easy-TF_dev.jsonl')
        middle_file = os.path.join(self.benchmark_dir, 'difficulty-mid-choice_dev.jsonl')

        def normalize(x):
            return str(x).strip()

        few_shot_prompt_dict = defaultdict(lambda: '')
        count_shots = defaultdict(lambda: 0)

        easy_data_list = self._load_jsonl(easy_file)
        for data in easy_data_list:
            c = data['id'].split('-')[0]
            if count_shots[c] == n_shot:
                continue

            few_shot_prompt_dict[c] += (
                f"سؤال: {normalize(data['question'])}"
                + "\n" +"إجابة: "+f"{normalize(data['answer'])}\n\n"
            )
            count_shots[c] += 1
        
        middle_data_list = self._load_jsonl(middle_file)
        for data in middle_data_list:
            c = data['id'].split('-')[0]
            if count_shots[c] == n_shot:
                continue

            few_shot_prompt_dict[c] += (
                f"سؤال: {normalize(data['question'])}\nA. "+ f"{normalize(data['A'])}"+"\nB. "+ f"{normalize(data['B'])}" +"\nC. " +f"{normalize(data['C'])}" +"\nD. " +f"{normalize(data['D'])}"
                + "\n" +"إجابة: "+f"{normalize(data['answer'])}\n\n"
            )
            count_shots[c] += 1

        for c in self.subtasks:
            few_shot_prompt_dict[c] += "{test_question}\nإجابة:" 

        return few_shot_prompt_dict
    


