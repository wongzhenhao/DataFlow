from dataflow.utils.reasoning_utils.Prompts import AnswerGeneratorPrompt
from dataflow.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.utils.APIGenerator_request import APIGenerator_request
import yaml
import logging
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils import Operator

from dataflow.utils.Storage import FileStorage
from dataflow.utils.Operator import Operator
from dataflow.utils.utils import init_model
@GENERATOR_REGISTRY.register()
class AnswerGenerator(Operator):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.prompt = AnswerGeneratorPrompt()
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.output_text_key = self.config.get("output_key", "response")
        self.input_key = self.config.get("input_key")
        self.output_key = self.config.get("output_key")
        self.logger = get_logger()

        self.generator = init_model(config)
        self.datastorage = FileStorage(config)

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'input_key']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    
    
    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于生成数学问题的标准答案，调用大语言模型进行分步推理和计算。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- output_file：输出文件路径\n"
                "- generator_type：生成器类型（aisuite/request）\n"
                "- model_name：使用的大模型名称\n"
                "- max_worker：并发线程数\n\n"
                "输出参数：\n"
                "- output_key：生成的答案字段"
            )
        elif lang == "en":
            return (
                "This operator generates standard answers for math problems using LLMs "
                "for step-by-step reasoning and calculation.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- output_file: Output file path\n"
                "- generator_type: Generator type (aisuite/request)\n"
                "- model_name: Name of the model used\n"
                "- max_worker: Number of threads\n\n"
                "Output Parameters:\n"
                "- output_key: Generated answer field"
            )
        else:
            return "AnswerGenerator produces standardized answers for mathematical questions."

    def run(self):
        '''
        Runs the answer generation process, reading from the input file and saving results to output.
        '''
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        self._validate_dataframe(dataframe)
        user_prompts = dataframe[self.input_key].tolist()
        answers = self.generator.generate_text_from_input(user_prompts)

        dataframe[self.output_text_key] = answers
        self.datastorage.write(self.output_file, dataframe)

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        '''
        Helper method to validate the input dataframe columns.
        '''
        if self.input_key not in dataframe.columns:
            raise ValueError(f"input_key: {self.input_key} not found in the dataframe.")
        
        if self.output_text_key in dataframe.columns:
            raise ValueError(f"Found {self.output_text_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
