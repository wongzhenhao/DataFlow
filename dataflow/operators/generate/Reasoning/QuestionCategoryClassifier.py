import json
import os
import pandas as pd
from dataflow.generator.utils import APIGenerator_aisuite, APIGenerator_request
from dataflow.generator.utils.Prompts import QuestionCategoryPrompt
import re
from dataflow.utils.registry import GENERATOR_REGISTRY
import logging
from dataflow.utils.utils import get_logger
from dataflow.generator.utils.CategoryFuzzer import normalize_categories, category_hasher, category_hasher_reverse
from dataflow.utils import Operator

@GENERATOR_REGISTRY.register()
class QuestionCategoryClassifier(Operator):
    def __init__(self, config: dict):
        """
        Initialize the QuestionCategoryClassifier with the provided configuration.
        """
        self.check_config(config)
        self.config = config
        self.prompts = QuestionCategoryPrompt()
        self.input_file = config.get("input_file")
        self.output_file= config.get("output_file")
        self.input_key = self.config.get("input_key", "question")
        self.output_key = self.config.get("output_key", "classification_result")  # default output key
        self.logger = get_logger()

        # Ensure input_file and output_file are provided
        if not self.input_file or not self.output_file:
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
        for text in dataframe[self.input_key]:
            used_prompt = self.prompts.question_synthesis_prompt(text)
            formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def _load_input(self):
        return pd.read_json(self.input_file, lines=True)

    def _write_output(self, save_path, dataframe, extractions):
        output_dir = os.path.dirname(save_path)
        os.makedirs(output_dir, exist_ok=True)
        dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

    def run(self):
        """
        Run the question category classification process.
        """
        # Read the input file
        # dataframe = pd.read_json(self.input_file, lines=True)
        dataframe = self._load_input()

        # Reformat the prompts for classification
        formatted_prompts = self._reformat_prompt(dataframe)

        # Generate responses using the model
        responses = self.model.generate_text_from_input(formatted_prompts)

        for (idx, row), classification_str in zip(dataframe.iterrows(), responses):
            try:
                if not classification_str:
                    raise ValueError("空字符串")

                # 去除 Markdown 代码块包裹
                cleaned_str = re.sub(r"^```json\s*|\s*```$", "", classification_str.strip(), flags=re.DOTALL)

                # 去除非 ASCII 字符（可选）
                cleaned_str = re.sub(r"[^\x00-\x7F]+", "", cleaned_str)

                classification = json.loads(cleaned_str)

                primary_raw = classification.get("primary_category", 10)
                secondary_raw = classification.get("secondary_category", 90)

                category_info = normalize_categories(primary_raw, secondary_raw)

                dataframe.at[idx, "primary_category"] = category_info["primary_category"]
                dataframe.at[idx, "secondary_category"] = category_info["secondary_category"]

            except json.JSONDecodeError:
                self.logger.warning(f"[警告] JSON 解析失败，收到的原始数据: {repr(classification_str)}")
            except Exception as e:
                self.logger.error(f"[错误] 解析分类结果失败: {e}")
                self.logger.debug(f"[DEBUG] 原始字符串：{repr(classification_str)}")


        if self.output_key in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key} in the dataframe, which leads to overwriting the existing column, please check the output_text_key: {key_list}")
        

        # Save DataFrame to the output file
        # dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)
        self._write_output(self.output_file, dataframe, None)

        self.logger.info(f"Classification results saved to {self.output_file}")
        return

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于对用户问题进行多级分类（主分类和子分类）。"
                "通过大语言模型对输入问题进行语义分析，输出分类编码结果。\n\n"
                "输入参数：\n"
                "- db_port/db_name/table_name：数据库连接参数（存储模式）\n"
                "- input_file/output_file：文件路径（文件模式）\n"
                "- input_key：输入数据中问题字段的键名\n"
                "- generator_type：模型调用方式（aisuite/request）\n\n"
                "输出参数：\n"
                "- classification_result：包含主分类和子分类的编码结果"
            )
        elif lang == "en":
            return (
                "Performs hierarchical classification (primary and secondary) on user questions. "
                "Utilizes LLM for semantic analysis and outputs category codes.\n\n"
                "Input Parameters:\n"
                "- db_port/db_name/table_name: Database connection params (storage mode)\n"
                "- input_file/output_file: File paths (file mode)\n"
                "- input_key: Key for question field in input data\n"
                "- generator_type: Model invocation method (aisuite/request)\n\n"
                "Output Parameters:\n"
                "- classification_result: Combined category code"
            )
