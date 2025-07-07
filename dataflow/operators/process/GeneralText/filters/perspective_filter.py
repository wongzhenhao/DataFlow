import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval.GeneralText import PerspectiveScorer

@OPERATOR_REGISTRY.register()
class PerspectiveFilter(OperatorABC):

    def __init__(self, min_score: float = 0.0, max_score: float = 0.1, url: str = 'https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = PerspectiveScorer(url=url)
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PerspectiveScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        # Get the scores for filtering
        scores = np.array(self.scorer.eval(dataframe, self.input_key))

        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
