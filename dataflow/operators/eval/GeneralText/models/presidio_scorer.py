from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage

# Presidio PII detection Scorer
@OPERATOR_REGISTRY.register()
class PresidioScorer(OperatorABC):
    def __init__(self, device='cuda', lang='en', model_cache_dir='./dataflow_cache'):
        self.language = lang
        self.device = device
        self.model_cache_dir = model_cache_dir
        self.model_name = 'dslim/bert-base-NER'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.logger = get_logger()
        warnings.filterwarnings("ignore", category=UserWarning, module="spacy_huggingface_pipelines")
        model_config = [{
            "lang_code": self.language,
            "model_name": {
                "spacy": "en_core_web_sm",
                "transformers": self.model_name
            }
        }]
        
        self.nlp_engine = TransformersNlpEngine(models=model_config)
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine)
        self.score_name = 'PresidioScore'
        self.logger.info(f"Presidio analyzer load complete!")


    def eval(self, dataframe, input_key):
        input_texts = dataframe.get(input_key, '').to_list()
        results = []
        for text in input_texts:
            analysis_results = self.analyzer.analyze(text=text, language=self.language)
            print(analysis_results)
            pii_count = len(analysis_results)
            results.append(pii_count)
        return results
    
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='PresidioScore'):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)        
        # Write the results to the output key in the dataframe
        dataframe[output_key] = scores
        storage.write(dataframe)
