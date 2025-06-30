from openai import OpenAI
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
# Alpagasus instruction quality evaluation
# cited from: AlpaGasus: Training A Better Alpaca with Fewer Data
@OPERATOR_REGISTRY.register()
class AlpagasusScorer(OperatorABC):
    def __init__(self, API_key = None, url = None, model = 'gpt-3.5-turbo', dimension = 'quality'):
        self.api_key = API_key
        self.url = url
        self.model = model
        self.dimension = dimension
        self.logger = get_logger()
        self.score_name = 'AlpagasusScore' 
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)

        self.system_prompt_template = """
        We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following.
        Instruction: {instruction}
        Input: {input}
        Response: {response}
        """
        self.user_prompt_template = """
        Please rate according to the {dimension} of the response to the instruction and the input. Each assistant
        receives a score on a scale of 0 to 5, where a higher score indicates a higher level of the {dimension}. Please
        first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.
        """

    def get_score(self, sample, input_instruction_key, input_input_key, input_output_key):
        instruction = sample.get(input_instruction_key, [''])
        response = sample.get(input_output_key, [''])
        input_text = sample.get(input_input_key, [''])
        system_prompt = self.system_prompt_template.format(instruction=instruction, input=input_text, response=response)
        user_prompt = self.user_prompt_template.format(dimension=self.dimension)
        api_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        score_line = api_response.choices[0].message.content.strip().split("\n")[0]
        
        score = float(score_line.split()[0])

        return score
    
    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str, input_input_key: str, input_output_key: str):
        scores = [
            self.get_score(sample, input_instruction_key, input_input_key, input_output_key)
            for sample in tqdm(dataframe.to_dict(orient='records'), desc="AlpagasusScorer Evaluating...")
        ]
        return scores

    def run(self, storage: DataFlowStorage, input_instruction_key: str, input_input_key: str, input_output_key: str, output_key: str):
        self.input_instruction_key = input_instruction_key
        self.input_input_key = input_input_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"NgramScore ready to evaluate, ")
        scores = self.eval(dataframe, self.input_instruction_key, self.input_input_key, self.output_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)