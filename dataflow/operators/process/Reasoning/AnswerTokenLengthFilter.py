from dataflow.core import ReasonerFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
from transformers import AutoTokenizer
from dataflow.utils.utils import get_logger
from datasets import Dataset
from dataflow.data import TextDataset
from dataflow.utils import Operator

@PROCESSOR_REGISTRY.register()
class AnswerTokenLengthFilter(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.filter_name = 'AnswerTokenLengthFilter'
        self.max_answer_token_length = config['max_answer_token_length']
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_dir'])
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
        def get_token_count(input_string):
            tokens = self.tokenizer.encode(input_string, add_special_tokens=False)
            return len(tokens)

        return np.array([get_token_count(item[self.keys]) <= self.max_answer_token_length for item in dataset]).astype(int)

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子根据token数量过滤过长的答案。\n\n"
                "输入参数：\n"
                "- max_answer_token_length：最大token数\n"
                "- tokenizer_dir：分词器路径\n"
                "- read_min/max_score：分数范围\n\n"
                "输出参数：\n"
                "- 长度合规返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "Filters answers exceeding specified token length limit.\n\n"
                "Input Parameters:\n"
                "- max_answer_token_length: Maximum allowed tokens\n"
                "- tokenizer_dir: Tokenizer directory\n"
                "- read_min/max_score: Score range\n\n"
                "Output Parameters:\n"
                "- Returns 1 if within limit, 0 otherwise"
            )
        else:
            return "AnswerTokenLengthFilter enforces answer length constraints"