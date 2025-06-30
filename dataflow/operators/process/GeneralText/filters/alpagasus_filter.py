from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval.GeneralText import AlpagasusScorer

@OPERATOR_REGISTRY.register()
class AlpagasusFilter(OperatorABC):

    def __init__(self, min_score=3, max_score=5, API_key = None, url = None, model='gpt-3.5-turbo', dimension='quality'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = AlpagasusScorer(API_key, url, model, dimension)

    def run(self, storage: DataFlowStorage, input_instruction_key: str, input_input_key: str, input_output_key: str, output_key: str='alpagasus_filter_label'):
        self.input_instruction_key = input_instruction_key
        self.input_input_key = input_input_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_instruction_key, self.input_input_key, self.input_output_key)
        print(scores)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
        
        