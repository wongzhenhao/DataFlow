from dataflow.core import ReasonerFilter
import numpy as np
import re
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.Eval.Text import NgramScorer
from dataflow.utils.utils import get_logger
from datasets import Dataset
from dataflow.data import TextDataset
from dataflow.utils import Operator

@PROCESSOR_REGISTRY.register()
class AnswerNgramFilter(Operator):
    def __init__(self, config: dict):
        self.config = config
        self.filter_name = 'AnswerNgramFilter'
        self.min_score = config['min_score']
        self.max_score = config['max_score']
        self.ngrams = config['ngrams']
        self.logger = get_logger()
        if "db_name" in config.keys():
            self.read_min_score: list = config['read_min_score']
            self.read_max_score: list = config['read_max_score']
            self.eval_stage = config['eval_stage']
            self.stage = config["stage"]
            self.pipeline_id = config["pipeline_id"]
            self.dataset = self._load_input()

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'generator_type']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
    def _load_input(self):
        pass
        
    def _write_output(self, labels, ids):
        pass

    def run(self, dataset):
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