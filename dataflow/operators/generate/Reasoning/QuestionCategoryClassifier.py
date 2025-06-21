import json
import os
import pandas as pd
from dataflow.generator.utils import APIGenerator_aisuite, APIGenerator_request
from dataflow.generator.utils.Prompts import QuestionCategoryPrompt
import re
from dataflow.utils.registry import GENERATOR_REGISTRY
import logging
from dataflow.utils.utils import get_logger
from dataflow.data import MyScaleStorage
from dataflow.generator.utils.CategoryFuzzer import normalize_categories, category_hasher, category_hasher_reverse

@GENERATOR_REGISTRY.register()
class QuestionCategoryClassifier():
    def __init__(self, args):
        """
        Initialize the QuestionCategoryClassifier with the provided configuration.
        """
        self.config = args
        self.prompts = QuestionCategoryPrompt()
        if "db_name" in args.keys():
            self.storage = MyScaleStorage(args['db_port'], args['db_name'], args['table_name'])
            self.input_file = None
            self.output_file= None
            self.stage = args.get("stage",0)
            self.pipeline_id = args.get("pipeline_id","")
            self.read_min_score = self.config.get('read_min_score', 0.9)
            self.read_max_score = self.config.get('read_max_score', 2.0)
            self.eval_stage = self.config.get('eval_stage', 2)
            self.read_format = self.config.get('read_format', '')
            self.read_syn = self.config.get('read_syn', '')
        else:
            self.input_file = args.get("input_file")
            self.output_file= args.get("output_file")
        self.read_key = self.config.get("read_key", "question")
        self.input_key = self.config.get("input_key", "data")  # default key for question input
        self.output_key = self.config.get("output_key", "classification_result")  # default output key
        self.logger = get_logger()

        # Ensure input_file and output_file are provided
        if not hasattr(self, "storage") and (not self.input_file or not self.output_file):
            raise ValueError("Both input_file and output_file must be specified in the config.")

        # Initialize the model
        self.model = self.__init_model__()

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
        # Check if read_key is in the dataframe
        if self.read_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"read_key: {self.read_key} not found in the dataframe. Available keys: {key_list}")

        formatted_prompts = []
        for text in dataframe[self.read_key]:
            used_prompt = self.prompts.question_synthesis_prompt(text)
            formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def _load_input(self):
        if hasattr(self, 'storage'):
            value_list = self.storage.read_json(
                [self.input_key], eval_stage=self.eval_stage, format=self.read_format, syn=self.read_syn, maxmin_scores=[{'max_score': self.read_max_score, 'min_score': self.read_min_score}], stage=self.stage, pipeline_id=self.pipeline_id, category="reasoning"
            )
            return pd.DataFrame([
                {**item['data'], 'id': str(item['id'])}
                for item in value_list
            ])
        else:
            return pd.read_json(self.input_file, lines=True)

    def _write_output(self, save_path, dataframe, extractions):
        if hasattr(self, 'storage'):
            output_rows = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
            output_1 = []
            for row in output_rows:
                try:
                    primary = row.get("primary_category")
                    secondary = row.get("secondary_category")
                    hash = category_hasher(primary, secondary)
                except:
                    hash = 170

                output_1.append({
                    "id": row.get("id"),
                    "category": hash,
                })
            self.storage.write_eval(output_1, algo_name="QuestionCategoryClassifier", score_key="category", stage=self.stage+1)
        else:
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
                "- read_key：输入数据中问题字段的键名\n"
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
                "- read_key: Key for question field in input data\n"
                "- generator_type: Model invocation method (aisuite/request)\n\n"
                "Output Parameters:\n"
                "- classification_result: Combined category code"
            )
