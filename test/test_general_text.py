from dataflow.operators.eval.GeneralText import VendiScorer 
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request
import os
os.environ['API_KEY'] = 'AIzaSyBzInL9rMlsVvovWoU7WV7U1OeLMkgYhCI'
class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.scorer = VendiScorer(device='cuda')

    def forward(self):
        self.scorer.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )

model = TextPipeline()
model.forward()
