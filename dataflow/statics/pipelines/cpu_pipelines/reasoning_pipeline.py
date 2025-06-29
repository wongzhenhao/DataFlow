from dataflow.operators.process.Reasoning import AnswerFormatterFilter
from dataflow.utils.storage import FileStorage

class ReasoningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
    
        self.answer_format_filter_step1 = AnswerFormatterFilter()
        
    def forward(self):
        self.answer_format_filter_step1.run(
            storage = self.storage.step(),
            input_key = "generated_cot",
        )

if __name__ == "__main__":
    model = ReasoningPipeline()
    model.forward()
