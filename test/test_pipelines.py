import pytest
from test_reasoning import ReasoningPipeline
from test_general_text import TextPipeline

@pytest.fixture(scope="session")
def llm_serving():
    from dataflow.llmserving import LocalModelLLMServing
    return LocalModelLLMServing(
        model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        tensor_parallel_size=4,
        max_tokens=8192,
        max_model_len=8192,
        model_source="local"
    )

@pytest.mark.gpu
def test_reasoning_pipeline(llm_serving): 
    pytest
    resoning_pipe = ReasoningPipeline(llm_serving=llm_serving)
    resoning_pipe.forward()

@pytest.mark.gpu
def test_text_pipeline(llm_serving):
    text_pipe = TextPipeline(llm_serving=llm_serving)
    text_pipe.forward()