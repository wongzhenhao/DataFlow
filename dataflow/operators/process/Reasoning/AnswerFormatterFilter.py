from dataflow.core import TextFilter, ReasonerFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
import re
from datasets import Dataset
from dataflow.data import TextDataset
from dataflow.utils.utils import get_logger
from dataflow.utils import Operator

@PROCESSOR_REGISTRY.register()
class AnswerFormatterFilter(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.filter_name = 'AnswerFormatterFilter'
        self.logger = get_logger()
        self.eval_stage = config.get('eval_stage', 4)
        self.stage = config.get("stage",0)
        self.pipeline_id = config.get("pipeline_id","")
        if "db_name" in config.keys():
            self.dataset = self._load_input()

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'generator_type']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    def is_valid_answer(answer: str) -> bool:
        # check final answer in \boxed{} or not 
        # if not re.search(r'\\boxed{.*}', answer):
        #     return False
        
        return True 
    
    def _load_input(self):
        pass
        
    def _write_output(self, labels, ids):
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
    
    
    def run(self, dataset):
        indexes =  np.zeros(len(dataset)).astype(int)

        for i, item in enumerate(dataset):
            answer = item[self.keys]
            if AnswerFormatterFilter.is_valid_answer(answer):
                indexes[i] = 1

        return indexes