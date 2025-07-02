from googleapiclient import discovery
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import pandas as pd
import os

@OPERATOR_REGISTRY.register()
class PerspectiveScorer(OperatorABC):
    def __init__(self, url: str='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1'):
        self.api_key = os.environ.get('API_KEY')
        self.api_name = 'commentanalyzer'
        self.api_version = 'v1alpha1'
        self.discovery_service_url = url
        self.logger = get_logger()
        self.logger.info('Initializing Perspective API...')
        # must enable VPN
        self.client = discovery.build(
            'commentanalyzer',
            'v1alpha1',
            developerKey=self.api_key,
            discoveryServiceUrl=self.discovery_service_url,
            static_discovery=False,
        )
        
        self.score_name = 'PerspectiveScore'
 

    def analyze_toxicity(self, text: str) -> float:
        """
        使用 Perspective API 分析文本的毒性得分
        """
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = self.client.comments().analyze(body=analyze_request).execute()
        return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']

    def get_score(self, samples, input_key: str) -> list:
        """
        为每个样本评估毒性分数
        """
        scores = []
        for sample in samples:
            text = sample.get(input_key, '')
            max_bytes = 20480  # 最大字节数限制
            
            if len(text.encode('utf-8')) > max_bytes:
                text = text.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
            
            score = self.analyze_toxicity(text)
            scores.append(score)

        return scores

    def eval(self, dataframe: pd.DataFrame, input_key: str) -> list:
        """
        为数据集中的每个样本批量评估毒性分数
        """
        # 将 dataframe 转换为字典列表
        samples = dataframe.to_dict(orient='records')
        scores = self.get_score(samples, input_key)
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PerspectiveScore'):
        """
        运行算子，评估并将分数写入存储
        """
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"PerspectiveScorer ready to evaluate.")
        scores = self.eval(dataframe, self.input_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
