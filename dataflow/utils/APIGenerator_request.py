import json
import requests
import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataflow.utils.Storage import FileStorage
from dataflow.utils.Generator import Generator
import re

class APIGenerator_request(Generator):
    def __init__(self, config: dict):
        self.config = config
        
        # Get API key from environment variable or config
        self.api_url = self.config.get("api_url", "https://api.openai.com/v1/chat/completions")
        self.api_key = os.environ.get("API_KEY")
        if self.api_key is None:
            raise ValueError("Lack of API_KEY")

        self.datastorage = FileStorage(self.config)
    
    def check_config(self):
        # Ensure all necessary keys are in the config
        necessary_keys = ['input_file', 'output_file', 'input_key', 'output_key', 'max_workers']
        for key in necessary_keys:
            if key not in self.config:
                raise ValueError(f"Key {key} is missing in the config")
            
    def format_response(self, response: dict) -> str:    
        # check if content is formatted like <think>...</think>...<answer>...</answer>
        content = response['choices'][0]['message']['content']
        if re.search(r'<think>.*</think>.*<answer>.*</answer>', content):
            return content
        
        try:
            reasoning_content = response['choices'][0]["message"]["reasoning_content"]
        except:
            reasoning_content = ""
        
        if reasoning_content != "":
            return f"<think>{reasoning_content}</think>\n<answer>{content}</answer>"
        else:
            return content

        


    def api_chat(self, system_info: str, messages: str, model: str):
        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_info},
                    {"role": "user", "content": messages}
                ]
            })

            headers = {
                'Authorization': f"Bearer {self.api_key}",
                'Content-Type': 'application/json',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
            }
            # Make a POST request to the API
            response = requests.post(self.api_url, headers=headers, data=payload, timeout=60)
            if response.status_code == 200:
                response_data = response.json()
                return self.format_response(response_data)
            else:
                logging.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logging.error(f"API request error: {e}")
            return None

    def generate(self):
        self.check_config()
        # Read input file
        raw_dataframe = self.datastorage.read(self.config['input_file'], "dataframe")
        if self.config['input_key'] not in raw_dataframe.columns:
            raise ValueError(f"input_key: {self.config['input_key']} not found in the dataframe.")
        
        logging.info(f"Found {len(raw_dataframe)} rows in the dataframe")

        responses = [None] * len(raw_dataframe)  # 创建一个列表，确保结果顺序与输入数据一致

        # Use ThreadPoolExecutor to handle multiple requests concurrently
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            for idx, row in raw_dataframe.iterrows():
                futures.append(
                    executor.submit(
                        self.api_chat,
                        self.config['system_prompt'],
                        row[self.config['input_key']],
                        self.config['model_name'],
                    )
                )
            
            for idx, future in enumerate(as_completed(futures)):
                response = future.result()
                responses[idx] = response  # 将响应放到正确的索引位置，确保顺序一致

        raw_dataframe[self.config['output_key']] = responses
        self.datastorage.write(self.config['output_file'], raw_dataframe)
        return

    def generate_from_input(self, input: list[str]) -> list[str]:
        def api_chat_with_id(system_info: str, messages: str, model: str, id):
            try:
                payload = json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_info},
                        {"role": "user", "content": messages}
                    ]
                })

                headers = {
                    'Authorization': f"Bearer {self.api_key}",
                    'Content-Type': 'application/json',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
                }
                # Make a POST request to the API
                response = requests.post(self.api_url, headers=headers, data=payload, timeout=1800)
                if response.status_code == 200:
                    # logging.info(f"API request successful")
                    response_data = response.json()
                    # logging.info(f"API response: {response_data['choices'][0]['message']['content']}")
                    return id,self.format_response(response_data)
                else:
                    logging.error(f"API request failed with status {response.status_code}: {response.text}")
                    return id,None
            except Exception as e:
                logging.error(f"API request error: {e}")
                return id,None
        responses = [None] * len(input)
        
        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = [
                executor.submit(
                    api_chat_with_id,
                    self.config['system_prompt'],
                    question,
                    self.config['model_name'],
                    idx
                ) for idx, question in enumerate(input)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses