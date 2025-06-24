import logging
from vllm import LLM,SamplingParams
from huggingface_hub import snapshot_download
import pandas as pd
from dataflow.utils.Storage import FileStorage
from dataflow.core import LLMServingABC

class LocalModelLLMServing(LLMServingABC):
    '''
    A class for generating text using vllm, with model from huggingface or local directory
    '''
    def __init__(self, config: dict):
        configs = config

        device = configs.get("device", "cuda")
        model_name_or_path = configs.get("model_path", None)
        temperature = configs.get("temperature", 0.7)
        top_p = configs.get("top_p", 0.9)
        max_tokens = configs.get("max_tokens", 512)
        top_k = configs.get("top_k", 40)
        repetition_penalty = configs.get("repetition_penalty", 1.0)
        seed = configs.get("seed", 42)
        system_prompt = configs.get("system_prompt", "You are a helpful assistant")
        download_dir = configs.get("download_dir", "./ckpt/models/")
        max_model_len = configs.get("max_model_len", 4096)

        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        try:
            self.real_model_path = snapshot_download(
                repo_id=model_name_or_path,
                local_dir=f"{download_dir}{model_name_or_path}",
            )
        except:
            self.real_model_path = model_name_or_path
        logging.info(f"Model will be loaded from {self.real_model_path}")
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed
            # hat_template_kwargs={"enable_thinking": False},
        )
        self.llm = LLM(
            model=self.real_model_path,
            device=device,
            max_model_len=max_model_len,
        )
        self.system_prompt = system_prompt
        self.datastorage = FileStorage(config)

    def generate(self):
        # read input file : accept jsonl file only
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        # check if input_key are in the dataframe
        if self.input_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"input_key: {self.input_key} not found in the dataframe, please check the input_key: {key_list}")
        # check if output_key are in the dataframe
        if self.output_key in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        # generate text
        user_prompts = dataframe[self.input_key].tolist()
        full_prompts = [self.system_prompt + '\n' + user_prompt for user_prompt in user_prompts]
        responses = self.llm.generate(full_prompts, self.sampling_params)
        dataframe[self.output_key] = [output.outputs[0].text for output in responses]
        self.datastorage.write(self.output_file, dataframe)
        return
    
    
    def generate_from_input(self,questions: list[str]) -> list[str]:
        full_prompts = [self.system_prompt + '\n' + question for question in questions]
        responses = self.llm.generate(full_prompts, self.sampling_params)
        return [output.outputs[0].text for output in responses]