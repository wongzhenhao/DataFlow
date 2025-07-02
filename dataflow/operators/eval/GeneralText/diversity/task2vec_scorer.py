from .task2vec.task2vec import Task2Vec
from .task2vec import task_similarity
import torch
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from torch.utils.data import Dataset
from dataflow import get_logger
from typing import Optional
# Task2Vec dataset diversity evaluation
# Cited from: Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data
@OPERATOR_REGISTRY.register()
class Task2VecScorer(OperatorABC):
    def __init__(self, device='cuda', sample_nums=10, sample_size=1, method: Optional[str]='montecarlo', model_cache_dir='./dataflow_cache'):
        self.sample_nums = sample_nums  # 样本数量
        self.sample_size = sample_size  # 每次采样的样本大小
        self.device = device
        self.model_cache_dir = model_cache_dir  
        self.score_name = 'Task2VecScore'
        self.method = method
        if method not in ['montecarlo', 'variational']:
            raise ValueError(f"Invalid method '{method}'. Valid options are 'montecarlo' and 'variational'.")
        self.logger = get_logger()

        # 初始化 tokenizer 和模型
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=self.model_cache_dir)
        self.probe_network = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=self.model_cache_dir)
        
        # 设置设备
        self.device = torch.device(self.device if self.device and torch.cuda.is_available() else "cpu")
        self.probe_network = self.probe_network.to(self.device)

    def preprocess(self, texts):
        """ 预处理文本数据，将其转换为 token ID """
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_outputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in tokenized_outputs.items()}

    def get_score(self, sentences):
        """ 获取多样性评分 """
        embeddings = []
        data_length = len(sentences)
        for sample_num in range(self.sample_nums):
            self.logger.info(f'--> Sample {sample_num + 1}/{self.sample_nums}')

            # 随机选取样本
            indices = random.sample(range(data_length), self.sample_size)
            texts = [sentences[i] for i in indices]
            tokenized_batch = self.preprocess(texts)

            # 嵌入并计算 Task2Vec
            tokenized_dataset = CustomTensorDataset(tokenized_batch)
            embedding, _ = Task2Vec(self.probe_network, method=self.method).embed(tokenized_dataset)
            embeddings.append(embedding)

        # 计算距离矩阵
        distance_matrix = task_similarity.pdist(embeddings, distance='cosine')
        div_coeff, conf_interval = task_similarity.stats_of_distance_matrix(distance_matrix)
        
        return {
            "Task2VecDiversityScore": div_coeff,
            "ConfidenceInterval": conf_interval
        }

    def eval(self, dataframe, input_key: str):
        """ 从 DataFrame 获取文本并计算评分 """
        samples = dataframe[input_key].to_list()
        # 获取分数
        task2vec_score = self.get_score(samples)
        self.logger.info(f"Task2Vec Diversity Score: {task2vec_score}")
        return task2vec_score

    def run(self, storage: DataFlowStorage, input_key: str):
        """ 从 DataFlowStorage 读取数据并计算评分 """
        dataframe = storage.read("dataframe")
        samples = dataframe[input_key].to_list()
        # 获取分数
        task2vec_score = self.get_score(samples)
        self.logger.info(f"Task2Vec Diversity Score: {task2vec_score}")
        return task2vec_score


class CustomTensorDataset(Dataset):
    def __init__(self, tokenized_batch):
        self.tokenized_batch = tokenized_batch

    def __getitem__(self, index):
        return {key: self.tokenized_batch[key][index] for key in self.tokenized_batch}

    def __len__(self):
        return len(next(iter(self.tokenized_batch.values())))
