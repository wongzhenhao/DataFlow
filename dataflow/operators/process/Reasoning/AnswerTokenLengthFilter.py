from dataflow.core import ReasonerFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
from transformers import AutoTokenizer
from dataflow.utils.utils import get_logger
from datasets import Dataset
from dataflow.data import TextDataset


@PROCESSOR_REGISTRY.register()
class AnswerTokenLengthFilter(ReasonerFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'AnswerTokenLengthFilter'
        self.max_answer_token_length = args_dict['max_answer_token_length']
        self.tokenizer = AutoTokenizer.from_pretrained(args_dict['tokenizer_dir'])
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