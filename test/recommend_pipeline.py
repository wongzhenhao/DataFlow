import pytest
from dataflow.operators.generate.Reasoning.QuestionGenerator import QuestionGenerator
from dataflow.operators.process.Reasoning.QuestionFilter import QuestionFilter
from dataflow.operators.generate.Reasoning.QuestionDifficultyClassifier import QuestionDifficultyClassifier
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request, LocalModelLLMServing


class RecommendPipeline():
    def __init__(self):

        # -------- FileStorage (请根据需要修改参数) --------
        self.storage = FileStorage(
            first_entry_file_name="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )


        # -------- LLM Serving (Local) --------
        llm_serving = LocalModelLLMServing(
            model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
            max_tokens=8192,
            model_source="local",
        )

        self.questiongenerator = QuestionGenerator(num_prompts=1, llm_serving=llm_serving)
        self.questionfilter = QuestionFilter(system_prompt="You are a helpful assistant.", llm_serving=llm_serving)
        self.questiondifficultyclassifier = QuestionDifficultyClassifier(llm_serving=llm_serving)

    def forward(self):
        self.questiongenerator.run(
            storage=self.storage.step(), input_key="instruction"
        )
        self.questionfilter.run(
            storage=self.storage.step(), input_key="output"
        )
        self.questiondifficultyclassifier.run(
            storage=self.storage.step(), input_key="output", output_key="difficulty_score"
        )


if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.forward()