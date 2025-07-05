from dataflow.operators.eval.GeneralText import LexicalDiversityScorer 
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request
import os
class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.scorer = LexicalDiversityScorer()

    def forward(self):
        self.scorer.run(
            storage=self.storage.step(),
            input_key='raw_content'
        )

model = TextPipeline()
model.forward()
