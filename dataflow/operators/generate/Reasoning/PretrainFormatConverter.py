from dataflow.prompts.reasoning import QuestionSynthesisPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow import get_logger

import pandas as pd
import random
import os

@OPERATOR_REGISTRY.register()
class PretrainFormatConverter():
    def __init__(self):
        """
        Initialize the Pretrain_FormatConvert_sft2pt with the provided configuration.
        """
        self.input_key = self.config.get("input_key", "data")
        self.read_key_question = self.config.get("read_key_question", "question")  # default key for question input
        self.read_key_answer = self.config.get("read_key_answer", "answer")  # default key for question input
        self.output_key = self.config.get("output_key", "text")  # default output key
        self.logger = get_logger()

    def run(self):
        """
        Run the pretrain data format convertion.
        """
        # Read the input
        dataframe = self._load_input()

        if self.output_key in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key} in the dataframe, which leads to overwriting the existing column, please check the output_text_key: {key_list}")
        
        # Save DataFrame to the output
        self._write_output(self.output_file, dataframe, None)

        self.logger.info(f"SFT to PT convertion results saved to {self.output_file}")
        return

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于将SFT格式数据转换为预训练格式。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- db_port/db_name：数据库连接配置\n"
                "- table_name：存储表名\n"
                "- eval_stage：数据处理阶段标识\n\n"
                "输出参数：\n"
                "- output_file：输出文件路径\n"
                "- 数据库存储：转换后的预训练格式数据"
            )
        elif lang == "en":
            return (
                "Converts SFT format data to pretraining format.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- db_port/db_name: Database connection config\n"
                "- table_name: Storage table name\n"
                "- eval_stage: Data processing stage\n\n"
                "Output Parameters:\n"
                "- output_file: Output file path\n"
                "- Database storage: Converted pretraining data"
            )
        else:
            return "FormatConvert_SFT_to_Pretrain: SFT to Pretraining format converter"