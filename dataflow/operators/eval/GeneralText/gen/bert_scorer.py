from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import evaluate

@OPERATOR_REGISTRY.register()
class BERTScorer(OperatorABC):
    def __init__(self, lang='en', model_cache_dir='./dataflow_cache'):
        self.batch_size =  1  # Default batch size
        self.data_type = "text"
        self.scorer_name = "BERTScoreScorer"
        self.score_type = float
        self.logger = get_logger()
        # Additional parameters for BERTScore
        self.lang = lang
        self.model_type = "distilbert-base-uncased"
        self.idf = False
        self.rescale_with_baseline = False
        self.bertscore = evaluate.load("bertscore", cache_dir=model_cache_dir)


    def eval(self, dataframe, input_key, reference_key):
        eval_data = dataframe[input_key].to_list()
        ref_data = dataframe[reference_key].to_list()

        if ref_data is None:
            raise ValueError("Reference data must be provided for BERTScore Scorer.")

        # Compute BERTScore for predictions and references
        results = self.bertscore.compute(
            predictions=eval_data,
            references=ref_data,
            lang=self.lang,
            model_type=self.model_type,
            idf=self.idf,
            rescale_with_baseline=self.rescale_with_baseline
        )
        # Extract F1 scores for batch and return
        scores = results["f1"]
        return scores
    
    def run(self, storage: DataFlowStorage, input_key: str, reference_key: str, output_key: str='BertScore'):
        self.input_key = input_key
        self.reference_key = reference_key
        self.output_key = output_key
        
        dataframe = storage.read("dataframe")
        self.logger.info(f"BertScore ready to evaluate.")
        
        scores = self.eval(dataframe, input_key, reference_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
