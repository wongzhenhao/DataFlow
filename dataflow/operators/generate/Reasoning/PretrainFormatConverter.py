from dataflow.prompts.reasoning import QuestionSynthesisPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow import get_logger

import pandas as pd
import random
import os

@OPERATOR_REGISTRY.register()
class PretrainFormatConverter(OperatorABC):
    def __init__(self):
        self.logger = get_logger()

    def run(self,
            storage: DataFlowStorage,
            read_key_question: str = "question",
            read_key_answer: str = "answer",
            output_key: str = "text",
            )
        self.read_key_question = read_key_question
        self.read_key_answer = read_key_answer
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        
        output_rows = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
        output_1 = []
        for row in output_rows:
                cur_q = row.get(self.read_key_question) if row.get(self.read_key_question) is not None else ""
                cur_a = row.get(self.read_key_answer) if row.get(self.read_key_answer) is not None else ""
                output_1.append({
                    "id": row.get("id"),
                    "text": cur_q + "\n" + cur_a,
                })

        output_file = storage.write(dataframe)
        self.logger.info(f"SFT to PT convertion results saved to {output_file}")
        
        return [read_key_question, read_key_answer, output_key]

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