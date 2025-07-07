import pandas as pd
import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval.GeneralText import LangkitScorer

@OPERATOR_REGISTRY.register()
class LangkitFilter(OperatorABC):
    def __init__(self, 
                 min_scores: dict = {
                    "flesch_reading_ease": 0,
                    "automated_readability_index": 0,
                    "aggregate_reading_level": 1.0,
                    "syllable_count": 100,
                    "lexicon_count": 200,
                    "sentence_count": 5,
                    "character_count": 500,
                    "letter_count": 400,
                    "polysyllable_count": 50,
                    "monosyllable_count": 150,
                    "difficult_words": 10,
                 },
                 max_scores: dict = {
                    "flesch_reading_ease": 10.0,
                    "automated_readability_index": 15.0,
                    "aggregate_reading_level": 12.0,
                    "syllable_count": 200,
                    "lexicon_count": 400,
                    "sentence_count": 50,
                    "character_count": 2000,
                    "letter_count": 1500,
                    "polysyllable_count": 100,
                    "monosyllable_count": 300,
                    "difficult_words": 50,
                 },
                 metrics_to_keep: list = [
                    "flesch_reading_ease",
                    "automated_readability_index",
                    "aggregate_reading_level",
                    "syllable_count",
                    "lexicon_count",
                    "sentence_count",
                    "character_count",
                    "letter_count",
                    "polysyllable_count",
                    "monosyllable_count",
                    "difficult_words",
                 ]):
        self.min_scores = min_scores
        self.max_scores = max_scores
        self.metric_name_map = {
            'flesch_reading_ease': 'LangkitFleschReadingEaseScore',
            'automated_readability_index': 'LangkitAutomatedReadabilityIndexScore',
            'aggregate_reading_level': 'LangkitAggregateReadingLevelScore',
            'syllable_count': 'LangkitSyllableCountScore',
            'lexicon_count': 'LangkitLexiconCountScore',
            'sentence_count': 'LangkitSentenceCountScore',
            'character_count': 'LangkitCharacterCountScore',
            'letter_count': 'LangkitLetterCountScore',
            'polysyllable_count': 'LangkitPolysyllableCountScore',
            'monosyllable_count': 'LangkitMonosyllableCountScore',
            'difficult_words': 'LangkitDifficultWordsScore'
        }
        if not self.min_scores.keys() == self.max_scores.keys():
            raise ValueError("min_scores and max_scores must have the same keys")  
        self.logger = get_logger()
        self.scorer = LangkitScorer()
        
    def run(self, storage: DataFlowStorage, input_key: str, output_keys: list = ["flesch_reading_ease", "automated_readability_index", "aggregate_reading_level", "syllable_count", "lexicon_count", "sentence_count", "character_count", "letter_count", "polysyllable_count", "monosyllable_count", "difficult_words"]):
        self.input_key = input_key
        self.output_keys = output_keys
        if not list(self.min_scores.keys()) == output_keys:
            raise ValueError("min_scores and output_keys must have the same keys")  
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        results = np.ones(len(dataframe), dtype=int)
        for _label in self.output_keys:
            label = self.metric_name_map[_label]
            min_score = self.min_scores[_label]
            max_score = self.max_scores[_label]
            dataframe[label] = pd.DataFrame(scores)[label]
            metric_scores = np.array(dataframe[label])
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            results = results & metric_filter.astype(int)
            self.logger.debug(f"Filtered by {_label}, {np.sum(results)} data remained")
            dataframe[f"{label}_label"] = metric_filter.astype(int)
        filtered_dataframe = dataframe[results == 1]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [f"{label}_label" for label in self.output_keys]

