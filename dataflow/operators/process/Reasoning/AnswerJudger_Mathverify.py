import pandas as pd
from tqdm import tqdm
from math_verify import parse, verify, LatexExtractionConfig
import logging
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils import Operator

@GENERATOR_REGISTRY.register()
class AnswerJudger_MathVerify(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.answer_key = self.config['answer_key']
        self.gt_key = self.config['gt_key']
        self.result_key = self.config['result_key']
        self.logger = get_logger()

    def check_config(self, config; dict) -> None:
        required_keys = [
            'input_file', 'output_file',
            'answer_key', 'gt_key', 'result_key',
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Key {key} is not in the config")
    
    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子通过符号计算验证答案正确性，执行数学表达式解析和等价性验证。\n\n"
                "输入参数：\n"
                "- answer_key：待验证答案字段名\n"
                "- gt_key：标准答案字段名\n"
                "- tolerance：数值容差阈值\n"
                "- symbolic_check：是否启用符号验证\n\n"
                "输出参数：\n"
                "- result_key：验证结果字段（True/False）"
            )
        elif lang == "en":
            return (
                "This operator verifies answer correctness through symbolic computation, "
                "performing mathematical expression parsing and equivalence checking.\n\n"
                "Input Parameters:\n"
                "- answer_key: Answer field to verify\n"
                "- gt_key: Ground truth field name\n"
                "- tolerance: Numerical tolerance threshold\n"
                "- symbolic_check: Enable symbolic verification\n\n"
                "Output Parameters:\n"
                "- result_key: Verification result field (True/False)"
            )
        else:
            return "AnswerJudger_MathVerify validates mathematical answer correctness."
        
    def _load_input(self):
        return pd.read_json(self.input_file, lines=True)

    def _write_output(self,save_path, dataframe, extractions):
        dataframe.to_json(save_path, orient="records", lines=True)

    def run(self):
        # raw_dataframe = pd.read_json(self.input_file, lines=True)
        raw_dataframe = self._load_input()
        key_list = raw_dataframe.columns.to_list()
        if self.answer_key not in key_list:
            raise ValueError(f"answer_key: {self.answer_key} not found in the dataframe, please check the input_key: {key_list}")
        if self.gt_key not in key_list:
            raise ValueError(f"gt_key: {self.gt_key} not found in the dataframe, please check the input_key: {key_list}")
        self.logger.info(f"Found {len(raw_dataframe)} rows in the dataframe")
        results = []
        for answer, gt in tqdm(zip(raw_dataframe[self.answer_key], raw_dataframe[self.gt_key]), total=len(raw_dataframe), desc='processed'):
            results.append(float(verify(parse(answer), parse(gt))) > 0)
        raw_dataframe[self.result_key] = results
        # raw_dataframe.to_json(self.output_file, orient='records', lines=True)
        self._write_output(self.output_file, raw_dataframe, None)
