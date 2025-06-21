import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from xVerify_Custom.src.xVerify.model import Model
from xVerify_Custom.src.xVerify.custommodel import Model_custom
from xVerify_Custom.src.xVerify.eval import Evaluator
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils import Operator

@GENERATOR_REGISTRY.register()
class AnswerJudger_Xverify(Operator):
    '''
    An algorithm to judge if the two answers are the same based on xVerify
    '''
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.logger = get_logger()

    def check_config(self, config: dict) -> None:
        """
        Check if all necessary keys are in the config dictionary
        """
        necessary_keys = [
            'input_file',
            'output_file',
            'question_key',
            'answer_1_key',
            'answer_2_key',
            'output_key',
            'inference_mode',
            'process_num'
        ]
        for key in necessary_keys:
            if key not in config:
                raise ValueError(f"The key {key} is not in the config")

    def load_model(self):
        """
        Load the model based on the inference_mode in config
        """
        inference_mode = self.config['inference_mode']
        model_name = self.config['model_name']
        model_path_or_url = self.config['model_path_or_url']
        api_key = self.config['api_key']

        if inference_mode == 'custom':
            # Initialize custom model
            model = Model_custom(
                model_name=model_name,
                model_path_or_url=model_path_or_url,
                inference_mode='api',  # Ensure inference mode for custom model
                api_key=api_key
            )
        else:
            # Initialize default model (API or local)
            model = Model(
                model_name=model_name,
                model_path_or_url=model_path_or_url,
                inference_mode=inference_mode,
                api_key=api_key
            )
        
        return model

    def _load_input(self):
        return pd.read_json(self.input_file, lines=True)

    def _write_output(self, save_path, dataframe, results):
        dataframe[self.config['output_key']] = results
        dataframe.to_json(save_path, orient='records', lines=True)

    def run(self):
        """
        Run the judging process on the input file, evaluating each row of data.
        """
        # Read the input data file
        # raw_dataframe = pd.read_json(self.config['input_file'], lines=True)
        raw_dataframe = self._load_input()
        
        # Check if the necessary keys exist in the dataframe
        for key in ['question_key', 'answer_1_key', 'answer_2_key']:
            if self.config[key] not in raw_dataframe.columns:
                raise ValueError(f"The key {self.config[key]} is not in the dataframe")

        # Check if output_key exists in the dataframe to avoid overwriting
        if self.config['output_key'] in raw_dataframe.columns:
            raise ValueError(f"The key {self.config['output_key']} already exists in the dataframe. Please choose another key.")

        # Load the model
        model = self.load_model()

        # Initialize the evaluator with the model and the number of processes
        evaluator = Evaluator(model=model, process_num=self.config['process_num'])

        results = []

        # If inference_mode is not 'custom', use a serial approach
        if self.config['inference_mode'] != 'custom':
            # Evaluate each row in the dataframe
            results = [
                evaluator.evaluate(
                    question=row[self.config['question_key']],
                    answer_1=row[self.config['answer_1_key']],
                    answer_2=row[self.config['answer_2_key']]
                )
                for _, row in raw_dataframe.iterrows()
            ]
        else:
            # For 'custom' mode, use ThreadPoolExecutor to process in parallel
            def process_row(row, index):
                result = evaluator.evaluate(
                    question=row[self.config['question_key']],
                    answer_1=row[self.config['answer_1_key']],
                    answer_2=row[self.config['answer_2_key']]
                )
                return result, index
            
            with ThreadPoolExecutor(max_workers=self.config['process_num']) as executor:
                futures = [
                    executor.submit(process_row, row, idx)
                    for idx, row in raw_dataframe.iterrows()
                ]
                
                # Collect the results in the correct order
                for future in futures:
                    result, index = future.result()
                    results.append((result, index))
                
                # Sort results by index to preserve order
                results.sort(key=lambda x: x[1])
                results = [result[0] for result in results]

        # Assign results to the output column and save to the output file
        self._write_output(self.config['output_file'], raw_dataframe, results)
        # raw_dataframe[self.config['output_key']] = results
        # raw_dataframe.to_json(self.config['output_file'], orient='records', lines=True)

        return
