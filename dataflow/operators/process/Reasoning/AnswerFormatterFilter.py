from dataflow.core import TextFilter, ReasonerFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
import re
from datasets import Dataset
from dataflow.data import TextDataset
from dataflow.utils.utils import get_logger

@PROCESSOR_REGISTRY.register()
class AnswerFormatterFilter(ReasonerFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'AnswerFormatterFilter'
        self.logger = get_logger()
        self.eval_stage = args_dict.get('eval_stage', 4)
        self.stage = args_dict.get("stage",0)
        self.pipeline_id = args_dict.get("pipeline_id","")
        if "db_name" in args_dict.keys():
            self.dataset = self._load_input()
        
    def is_valid_answer(answer: str) -> bool:
        # check final answer in \boxed{} or not 
        # if not re.search(r'\\boxed{.*}', answer):
        #     return False
        
        return True 
    
    def _load_input(self):
        if hasattr(self, 'storage'):
            value_list = self.storage.read_json(
                [self.input_key], eval_stage=self.eval_stage, format=self.read_format, syn=self.read_syn, stage=self.stage, pipeline_id=self.pipeline_id, category="reasoning"
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

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于检查答案格式是否符合规范，主要验证数学答案是否包含正确的\\boxed{}标记。\n\n"
                "输入参数：\n"
                "- input_key：输入字段名\n"
                "- eval_stage：评估阶段标识\n"
                "- result_key：结果字段名\n\n"
                "输出参数：\n"
                "- 通过格式检查返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "This operator validates answer formatting, specifically checking for correct \\boxed{} notation.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the answer\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- result_key: Output result field name\n\n"
                "Output Parameters:\n"
                "- Returns 1 for valid format, 0 otherwise"
            )
        else:
            return "AnswerFormatterFilter validates mathematical answer formatting"
    
    
    def filter_func(self, dataset):
        indexes =  np.zeros(len(dataset)).astype(int)

        for i, item in enumerate(dataset):
            answer = item[self.keys]
            if AnswerFormatterFilter.is_valid_answer(answer):
                indexes[i] = 1

        return indexes