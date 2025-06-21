import json
import os
import pandas as pd
from dataflow.generator.utils import APIGenerator_aisuite, APIGenerator_request
from dataflow.generator.utils.Prompts import QuestionDifficultyPrompt
import re
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils import Operator

@GENERATOR_REGISTRY.register()
class QuestionDifficultyClassifier(Operator):
    def __init__(self, config):
        """
        Initialize the QuestionCategoryClassifier with the provided configuration.
        """
        self.check_config(config)
        self.config = config
        self.prompts = QuestionDifficultyPrompt()
        self.input_file = config.get("input_file")
        self.output_file= config.get("output_file")
        self.input_key = self.config.get("input_key", "question")  # default key for question input
        self.output_key = self.config.get("output_key", "classification_result")  # default output key
        self.logger = get_logger()
        
        # Ensure input_file and output_file are provided
        if not hasattr(self,'input_file') or not hasattr(self,'output_file'):
            raise ValueError("Both input_file and output_file must be specified in the config.")

        # Initialize the model
        self.model = self.__init_model__()
    
    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'generator_type']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
    def __init_model__(self):
        """
        Initialize the model generator based on the configuration.
        """
        generator_type = self.config.get("generator_type", "local").lower()
        
        if generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        # Check if input_key is in the dataframe
        if self.input_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"input_key: {self.input_key} not found in the dataframe. Available keys: {key_list}")

        formatted_prompts = []
        for i, text in enumerate(dataframe[self.input_key]):
            if text is not None:
                used_prompt = self.prompts.question_synthesis_prompt(text)
            else:
                used_prompt = None
            formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def _load_input(self):
        return pd.read_json(self.input_file, lines=True)

    def _write_output(self, save_path, dataframe, extractions):
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        dataframe.to_json(save_path, orient="records", lines=True)

    def run(self):
        # read input file : accept jsonl file only
        # dataframe = pd.read_json(self.input_file,lines=True)
        dataframe = self._load_input()
        # model = self.__init_model__()
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.model.generate_text_from_input(formatted_prompts)

        rating_scores = []
        for response in responses:
            # 修改后的正则表达式匹配数字和小数点，同时过滤非法结尾
            match = re.search(r'Rating:\s*((\d+\.\d+)|\d+)', response)
            if match:
                score_str = match.group(1).rstrip('.')  # 去除末尾可能的小数点
                try:
                    score = float(score_str)
                except ValueError:
                    score = -1
            else:
                score = -1
            rating_scores.append(score)

        #if self.output_key in dataframe.columns:
        #    key_list = dataframe.columns.tolist()
        #    raise ValueError(f"Found {self.output_text_key} in the dataframe, which leads to overwriting the existing column, please check the output_text_key: {key_list}")
        
        dataframe[self.output_key] = rating_scores
        
        
            # Save DataFrame to the output file
        # dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)
        self._write_output(self.output_file, dataframe, None)

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于评估问题的难度等级。"
                "通过大语言模型分析问题复杂度，输出1-10级的难度评分。\n\n"
                "输入参数：\n"
                "- eval_stage：评估阶段标识\n"
                "- read_min/max_score：分数过滤阈值\n"
                "- 其他参数同QuestionCategoryClassifier\n\n"
                "输出参数：\n"
                "- difficulty_score：数值型难度评分（1-10）"
            )
        elif lang == "en":
            return (
                "Evaluates question difficulty level using LLM analysis. "
                "Outputs numerical difficulty score from 1 to 10.\n\n"
                "Input Parameters:\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- read_min/max_score: Score filtering thresholds\n"
                "- Other params same as QuestionCategoryClassifier\n\n"
                "Output Parameters:\n"
                "- difficulty_score: Numerical difficulty rating (1-10)"
            )