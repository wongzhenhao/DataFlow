import pytest
from dataflow.operators.process.Reasoning.AnswerPipelineRoot import AnswerPipelineRoot
from dataflow.operators.generate.Reasoning.PseudoAnswerGenerator import PseudoAnswerGenerator
from dataflow.operators.process.Reasoning.AnswerFormatterFilter import AnswerFormatterFilter
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

        self.answerpipelineroot = AnswerPipelineRoot()
        self.pseudoanswergenerator = PseudoAnswerGenerator(llm_serving=llm_serving, max_times=3)
        self.answerformatterfilter = AnswerFormatterFilter()

    def forward(self):
        self.answerpipelineroot.run(
            storage=self.storage.step(), input_answer_key="output", input_gt_key="golden_answer"
        )
        self.pseudoanswergenerator.run(
            storage=self.storage.step(), input_key="instruction", output_key_answer="pseudo_answers", output_key_answer_value="pseudo_answer_value", output_key_solutions="pseudo_solutions", output_key_correct_solution_example="pseudo_correct_solution_example"
        )
        self.answerformatterfilter.run(
            storage=self.storage.step(), input_key="generated_cot"
        )


if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.forward()
