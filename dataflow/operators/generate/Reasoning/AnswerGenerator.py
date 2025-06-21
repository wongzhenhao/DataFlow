from dataflow.generator.utils.Prompts import AnswerGeneratorPrompt
# from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
import yaml
import logging
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.data import MyScaleStorage

@GENERATOR_REGISTRY.register()
class AnswerGenerator:
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, config: dict):
        self.config = config
        self.prompt = AnswerGeneratorPrompt()
        self.model_generator = self.__init_model__()
        if "db_name" in config.keys():
            self.storage = MyScaleStorage(config['db_port'], config['db_name'], config['table_name'])
            self.input_file = None
            self.output_file= None
            self.stage = config.get("stage",0)
            self.pipeline_id = config.get("pipeline_id","")
            self.read_min_score: list = self.config.get('read_min_score', [])
            self.read_max_score: list = self.config.get('read_max_score', [])
            self.read_format = self.config.get('read_format', '')
            self.read_syn = self.config.get('read_syn', '')
            self.write_format = self.config.get('write_format', '')
            self.write_syn = self.config.get('write_syn', '')
            self.eval_stage = self.config.get('eval_stage', 4)
        else:
            self.input_file = self.config['input_file']
            self.output_file = self.config['output_file']
        # self.input_file = self.config.get("input_file")
        # self.output_file = self.config.get("output_file")
        self.input_key = self.config.get("input_key", "data")
        self.read_key = self.config.get("read_key", "prompt")
        self.output_text_key = self.config.get("output_key", "response")
        self.logger = get_logger()
        # Ensure required paths and keys are provided
        if not hasattr(self,"storage") and (not self.input_file or not self.output_file):
            raise ValueError("Both input_file and output_file must be specified in the config.")

    def __init_model__(self):
        '''
        Initialize the model generator based on the configuration.
        '''
        generator_type = self.config.get("generator_type", "local").lower()
        if generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")
    
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
        
    def _load_input(self):
        if hasattr(self, 'storage'):
            value_list = self.storage.read_json(
                [self.input_key], eval_stage=self.eval_stage, format=self.read_format, syn=self.read_syn, maxmin_scores=[dict(zip(['min_score', 'max_score'], list(_))) for _ in list(zip(self.read_min_score, self.read_max_score))], stage=self.stage, pipeline_id=self.pipeline_id, category="reasoning"
            )
            return pd.DataFrame([
                {**item['data'], 'id': str(item['id'])}
                for item in value_list
            ])
        else:
            return pd.read_json(self.input_file, lines=True)

    def _write_output(self,save_path, dataframe, extractions):
        if hasattr(self, 'storage'):
            # output_rows = []
            # for i, row in dataframe.iterrows():
            #     output_rows.append({
            #         self.read_key: row[self.read_key],
            #         self.output_text_key: row[self.output_text_key]
            #     })
            output_rows = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
            self.storage.write_data(output_rows, format=self.write_format, Synthetic=self.write_syn, stage=self.stage+1)
        else:
            dataframe.to_json(save_path, orient="records", lines=True)

    def run(self):
        '''
        Runs the answer generation process, reading from the input file and saving results to output.
        '''
        # Read input file: only accept jsonl format
        # dataframe = pd.read_json(self.input_file, lines=True)
        dataframe = self._load_input()
        # print(dataframe)
        # Ensure the input and output keys are correctly set
        self._validate_dataframe(dataframe)

        # Extract the prompts and generate answers
        user_prompts = dataframe[self.read_key].tolist()
        answers = self.model_generator.generate_text_from_input(user_prompts)

        # Save the generated answers to the output file
        dataframe[self.output_text_key] = answers
        # dataframe.to_json(self.output_file, orient="records", lines=True)
        self._write_output(self.output_file, dataframe, None)

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        '''
        Helper method to validate the input dataframe columns.
        '''
        # Check if the input prompt key exists in the dataframe
        if self.read_key not in dataframe.columns:
            raise ValueError(f"read_key: {self.read_key} not found in the dataframe.")
        
        # Check if the output text key already exists in the dataframe
        if self.output_text_key in dataframe.columns:
            raise ValueError(f"Found {self.output_text_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
