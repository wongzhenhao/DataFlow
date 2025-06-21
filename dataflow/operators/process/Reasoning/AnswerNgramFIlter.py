from dataflow.core import ReasonerFilter
import numpy as np
import re
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.Eval.Text import NgramScorer
from dataflow.utils.utils import get_logger
from datasets import Dataset
from dataflow.data import TextDataset


@PROCESSOR_REGISTRY.register()
class AnswerNgramFilter(ReasonerFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'AnswerNgramFilter'
        self.min_score = args_dict['min_score']
        self.max_score = args_dict['max_score']
        self.ngrams = args_dict['ngrams']
        self.logger = get_logger()
        if "db_name" in args_dict.keys():
            self.read_min_score: list = args_dict['read_min_score']
            self.read_max_score: list = args_dict['read_max_score']
            self.eval_stage = args_dict['eval_stage']
            self.stage = args_dict["stage"]
            self.pipeline_id = args_dict["pipeline_id"]
            self.dataset = self._load_input()

    def _load_input(self):
        if hasattr(self, 'storage'):
            value_list = self.storage.read_json(
                [self.input_key], eval_stage=self.eval_stage, format=self.read_format, syn=self.read_syn, maxmin_scores=[dict(zip(['min_score', 'max_score'], list(_))) for _ in list(zip(self.read_min_score, self.read_max_score))], stage=self.stage, pipeline_id=self.pipeline_id, category="reasoning"
            )
            value_list = [        
                {**item['data'], 'id': str(item['id'])}
                for item in value_list
            ]
            
            dataset = Dataset.from_list(value_list)
            return TextDataset(
                dataset=dataset,
                keys=value_list[0].keys(),
                metadata=None 
            )
        else:
            pass
        
    def _write_output(self, labels, ids):
        if hasattr(self, 'storage'):
            output_rows = []
            for _, label in zip(ids, labels):
                output_rows.append({
                    self.result_key: label,
                    'id': _
                })
            self.storage.write_eval(output_rows, algo_name=self.filter_name, score_key=self.result_key, stage=self.stage+1)
        else:
            pass

    def filter_func(self, dataset):
        scores = []
        for sample in dataset:
            answer = sample[self.question_key]
            try:
                answer += sample[self.answer_key]
            except:
                pass
            content = answer.lower()
            content = re.sub(r'[^\w\s]', '', content)
            words = content.split()
            ngrams = [' '.join(words[i:i + self.ngrams]) for i in range(len(words) - (self.ngrams - 1))]
            unique_ngrams = set(ngrams)

            total_ngrams = len(ngrams)
            unique_ngrams_count = len(unique_ngrams)

            repetition_score = unique_ngrams_count / total_ngrams if total_ngrams > 0 else 0.0
            scores.append(repetition_score) 

        return np.array([self.min_score <= score <= self.max_score for score in scores]).astype(int)

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子基于n-gram重复率过滤答案，检测回答中的重复模式。\n\n"
                "输入参数：\n"
                "- min_score：最小可接受分数\n"
                "- max_score：最大可接受分数\n"
                "- ngrams：n-gram大小\n\n"
                "输出参数：\n"
                "- 分数在范围内返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "This filter detects repetitive patterns using n-gram repetition scores.\n\n"
                "Input Parameters:\n"
                "- min_score: Minimum acceptable score\n"
                "- max_score: Maximum acceptable score\n"
                "- ngrams: Size of n-grams\n\n"
                "Output Parameters:\n"
                "- Returns 1 if score is within range, 0 otherwise"
            )
        else:
            return "AnswerNgramFilter detects answer repetition"