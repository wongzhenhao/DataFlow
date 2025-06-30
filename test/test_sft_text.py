 
from dataflow.operators.process.GeneralText import (

    AlpagasusFilter
)
from dataflow.utils.storage import FileStorage

class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/sft_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.alpagasus_filter = AlpagasusFilter(min_score=3,max_score=5,API_key=None,url=None)

    def forward(self):

        self.alpagasus_filter.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_input_key="input",
            input_output_key='output'
        )

model = TextPipeline()
model.forward()
