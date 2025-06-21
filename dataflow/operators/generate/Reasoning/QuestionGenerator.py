import random
import os
import pandas as pd
try:
    from utils import  APIGenerator_aisuite, APIGenerator_request
    from utils.Prompts import QuestionSynthesisPrompt
except ImportError:
    from dataflow.generator.utils import  APIGenerator_aisuite, APIGenerator_request
    from dataflow.generator.utils.Prompts import QuestionSynthesisPrompt
from dataflow.utils.registry import GENERATOR_REGISTRY
import logging
from dataflow.utils.utils import get_logger
from dataflow.utils import Operator 

@GENERATOR_REGISTRY.register()
class QuestionGenerator(Operator):
    def __init__(self, config):
        """
        Initialize the QuestionGenerator with the provided configuration.
        """
        self.check_config(config)
        self.config = config
        self.prompts = QuestionSynthesisPrompt()
        self.input_file = config.get("input_file")
        self.output_file= config.get("output_file")
        self.input_key = self.config.get("input_key", "question")  # default key for question input
        self.num_prompts = self.config.get("num_prompts", 1)  # default number of prompts to use for generation
        # check if num_prompts is a valid number
        if self.num_prompts not in [0,1,2,3,4,5]:
            raise ValueError("num_prompts must be 0, 1, 2, 3, 4, or 5")

        # Validate that input_file and output_file are provided
        if "db_name" not in config.keys() and (not self.input_file or not self.output_file):
            raise ValueError("Both input_file and output_file must be specified in the config.")

        # Initialize the model
        self.model = self.__init_model__()
        self.logger = get_logger()

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'generator_type']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")


    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于基于现有问题生成新问题。\n\n"
                "输入参数：\n"
                "- eval_stage：评估阶段标识\n"
                "- read_min/max_score：分数过滤阈值\n"
                "- 其他参数同基础分类器\n\n"
                "输出参数：\n"
                "- generated_questions：生成的新问题列表（每个原问题生成1-5个）"
            )
        elif lang == "en":
            return (
                "Generates new questions based on existing ones. "
                "Produces 1-5 new questions per original question.\n\n"
                "Input Parameters:\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- read_min/max_score: Score filtering thresholds\n"
                "- Other params same as base classifier\n\n"
                "Output Parameters:\n"
                "- generated_questions: List of newly generated questions"
            )

    def __init_model__(self):
        """
        Initialize the model generator based on the configuration.
        """
        generator_type = self.config.get("generator_type", "local").lower()

        # if generator_type == "local":
        #     return LocalModelGenerator(self.config)
        if generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions based on num_prompts.
        """
        # Check if input_key is in the dataframe
        if self.input_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"input_key: {self.input_key} not found in the dataframe. Available keys: {key_list}")

        # Predefined transformation options for diversity
        diversity_mode = [
            "1, 2, 3",
            "1, 2, 4",
            "1, 2, 5",
            "1, 4, 5",
            "1, 2, 3, 4, 5"
        ]

        formatted_prompts = []
        for question in dataframe[self.input_key]:
            if self.num_prompts == 0:
                formatted_prompts.append("")  # Skip generating for this question
            else:
                # Randomly choose the required number of transformations from diversity_mode
                selected_items = random.sample(diversity_mode, self.num_prompts)
                for selected_item in selected_items:
                    used_prompt = self.prompts.question_synthesis_prompt(selected_item, question)
                    formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def _load_input(self):
        return pd.read_json(self.input_file, lines=True)

    def _write_output(self, save_path, dataframe, extractions):
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        dataframe.to_json(save_path, orient="records", lines=True)

    def run(self):
        """
        Run the question generation process.
        """
        try:
            
            # Read the input file (jsonl format only)
            # dataframe = pd.read_json(self.input_file, lines=True)
            dataframe = self._load_input()
            if self.input_key not in dataframe.columns:
                raise ValueError(f"input_key: {self.input_key} not found in the dataframe. Available keys: {dataframe.columns.tolist()}")
            if "Synth_or_Input" in dataframe.columns:
                raise ValueError(f"Synth_or_Input is a reserved column name to show if the question is generated or not, please rename it")
            
            if self.num_prompts == 0:
                raise ValueError(f"num_prompts should not be 0.")
                # self.logger.info(f"num_prompts is 0, skip generation")
                # # dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)
                # self.logger.info(f"Generated questions saved to {self.output_file}")
                # return

            # Reformat the prompts for question generation
            formatted_prompts = self._reformat_prompt(dataframe)

            # Generate responses using the model
            responses = self.model.generate_text_from_input(formatted_prompts)

            # 将新生成的问题作为新的行添加到dataframe中，仍然填写到input_key中,这些行的其他列全部为空
            # new_rows = pd.DataFrame(columns=dataframe.columns)
            # new_rows[self.input_key] = responses
            repeat_count = self.num_prompts  # 每个seed生成几个
            expected_total = len(dataframe) * repeat_count

            if len(responses) != expected_total:
                raise ValueError(f"Expected {expected_total} responses (len(dataframe)={len(dataframe)} * {repeat_count}), but got {len(responses)}")

            new_rows = pd.DataFrame({
                self.input_key: responses,
            })
            
            new_rows["Synth_or_Input"] = "synth"
            dataframe["Synth_or_Input"] = "input"
            
            dataframe = pd.concat([dataframe, new_rows], ignore_index=True) # ignore_index=True 表示忽略原来的索引
        
            # Ensure output directory exists
            # output_dir = os.path.dirname(self.output_file)
            # os.makedirs(output_dir, exist_ok=True)

            # Save DataFrame to JSON file
            # 过滤self.input_key为空的数据
            dataframe = dataframe[dataframe[self.input_key].notna()]
            dataframe = dataframe[dataframe[self.input_key] != ""]
            # dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)
            self._write_output(self.output_file, dataframe, None)

            self.logger.info(f"Generated questions saved to {self.output_file}")

        except Exception as e:
            self.logger.error(f"[错误] 处理过程中发生异常: {e}")

    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于基于种子问题生成多样化的问题变体。"
                "通过多种语义转换方式扩展问题数据集。\n\n"
                "输入参数：\n"
                "- num_prompts：生成变体数量（0-5）\n"
                "- input_key：输入数据键名\n"
                "- 其他存储/模型参数同前\n\n"
                "输出参数：\n"
                "- Synth_or_Input：标识生成或原始问题\n"
                "- 生成的问题存储于指定字段"
            )
        elif lang == "en":
            return (
                "Generates diverse question variations from seed questions "
                "using multiple semantic transformation methods.\n\n"
                "Input Parameters:\n"
                "- num_prompts: Number of variations to generate (0-5)\n"
                "- input_key: Key for input data\n"
                "- Other storage/model params as previous\n\n"
                "Output Parameters:\n"
                "- Synth_or_Input: Marks generated vs original questions\n"
                "- Generated questions in specified field"
            )
