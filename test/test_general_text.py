from dataflow.operators.eval.GeneralText import RMScorer 
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request
import os
class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/sft_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.scorer = RMScorer(device='cuda')

    def forward(self):
        self.scorer.run(
            storage=self.storage.step()
        )

model = TextPipeline()
model.forward()
