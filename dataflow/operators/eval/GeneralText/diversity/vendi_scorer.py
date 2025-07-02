from vendi_score import text_utils
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

# VendiScore dataset diversity evaluation
# Cited from: The Vendi Score: A Diversity Evaluation Metric for Machine Learning
@OPERATOR_REGISTRY.register()
class VendiScorer(OperatorABC):
    def __init__(self, device='cuda'):
        self.bert_model_path = 'bert-base-uncased'
        self.simcse_model_path = 'princeton-nlp/unsup-simcse-bert-base-uncased'
        self.device = device
        self.score_name = 'VendiScore'
        self.logger = get_logger()

    def get_score(self, sentences):
        result = {}
        # ngram_vs = text_utils.ngram_vendi_score(sentences, ns=[1, 2, 3, 4])
        # result["N-gramsVendiScore"] = round(ngram_vs, 2)
        bert_vs = text_utils.embedding_vendi_score(sentences, model_path=self.bert_model_path, device=self.device)
        result["BERTVendiScore"] = round(bert_vs, 2)
        simcse_vs = text_utils.embedding_vendi_score(sentences, model_path=self.simcse_model_path, device=self.device)
        result["SimCSEVendiScore"] = round(simcse_vs, 2)
        return result
    
    def eval(self, dataframe, input_key: str):
        samples = dataframe[input_key].to_list()
        # 获取分数
        vendiscore = self.get_score(samples)
        self.logger.info(f"VendiScore: {vendiscore}")
        return vendiscore

    def run(self, storage: DataFlowStorage, input_key: str):
        dataframe = storage.read("dataframe")
        samples = dataframe[input_key].to_list()
        # 获取分数
        vendiscore = self.get_score(samples)
        self.logger.info(f"VendiScore: {vendiscore}")
        return vendiscore