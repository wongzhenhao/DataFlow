from dataflow.operators.eval.GeneralText import NgramScorer
from dataflow.operators.process.GeneralText import NgramFilter, MinHashDeduplicator
from dataflow.operators.generate.GeneralText import HtmlUrlRemoverRefiner
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request

class TextPipeline():
    def __init__(self):
        
        self.storage = FileStorage(
            first_entry_file_name="/mnt/public/data/mzm/DataFlow-Preview/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        self.ngram_scorer_step_1 = NgramFilter(min_score=0.99, max_score=1.0, ngrams=5)
        self.html_remover_step_2 = HtmlUrlRemoverRefiner()
        self.minhash_deduplicator_step_3 = MinHashDeduplicator(num_perm=128, threshold=0.9, use_n_gram=True, ngram=5)
        
    def forward(self):
        
        self.ngram_scorer_step_1.run(
            storage = self.storage.step(),
            input_key = "raw_content",
            output_key = "ngram_score",
        )
        
        self.html_remover_step_2.run(
            storage = self.storage.step(),
            input_keys=['raw_content'],
            output_key='html_removed_content',
        )
        
        self.minhash_deduplicator_step_3.run(
            storage = self.storage.step(),
            input_keys=['raw_content'],
            output_key='minhash_deduplicated_label',
        )

model = TextPipeline()
model.forward()