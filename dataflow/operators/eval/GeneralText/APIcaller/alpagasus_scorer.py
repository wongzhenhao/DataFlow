from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from tqdm import tqdm
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import AlpagasusPrompt  

@OPERATOR_REGISTRY.register()
class AlpagasusScorer(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, dimension: str = 'quality'):
        """
        初始化，修改为使用 LLMServing 实现
        """
        self.llm_serving = llm_serving
        self.logger = get_logger()
        self.score_name = 'AlpagasusScore'
        self.dimension = dimension
        
        self.prompt = AlpagasusPrompt(dimension=self.dimension)

    def get_score(self, samples, input_instruction_key, input_input_key, input_output_key):
        system_prompts = []
        user_prompts = []

        for sample in samples:
            instruction = sample.get(input_instruction_key, [''])
            response = sample.get(input_output_key, [''])
            input_text = sample.get(input_input_key, [''])

            # 生成 system 和 user prompts
            system_prompts.append(self.prompt.build_system_prompt(instruction, input_text, response))
            user_prompts.append(self.prompt.build_user_prompt())

        # 一次性传递所有生成的 prompts
        inputs = [system + "\n" + user for system, user in zip(system_prompts, user_prompts)]
        
        # 批量调用 LLMServing 生成响应
        responses = self.llm_serving.generate_from_input(user_inputs=inputs)
        
        # 提取得分
        scores = []
        for response in responses:
            score_line = response.strip().split("\n")[0]
            score = float(score_line.split()[0])  # 获取得分部分
            scores.append(score)
            
        return scores

    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str, input_input_key: str, input_output_key: str):
        """
        为数据集中的每个样本批量评估分数
        """
        # 将 dataframe 转换为字典列表
        samples = dataframe.to_dict(orient='records')

        # 批量处理并获取所有得分
        scores = self.get_score(samples, input_instruction_key, input_input_key, input_output_key)
        return scores


    def run(self, storage: DataFlowStorage, input_instruction_key: str, input_input_key: str, input_output_key: str, output_key: str):
        """
        运行算子，评估并将分数写入存储
        """
        self.input_instruction_key = input_instruction_key
        self.input_input_key = input_input_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"AlpagasusScorer ready to evaluate.")
        
        scores = self.eval(dataframe, self.input_instruction_key, self.input_input_key, self.input_output_key)
        
        # 将分数写入输出列
        dataframe[self.output_key] = scores
        storage.write(dataframe)
