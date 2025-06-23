from dataflow.operators.process.Reasoning.QuestionFilter import (
    QuestionFilter
)
from dataflow.operators.generate.Reasoning.QuestionGenerator import (
    QuestionGenerator
)
from dataflow.operators.generate.Reasoning.QuestionDifficultyClassifier import (
    QuestionDifficultyClassifier
)
from dataflow.operators.generate.Reasoning.QuestionCategoryClassifier import (
    QuestionCategoryClassifier
)
from dataflow.utils.Storage import FileStorage
from dataflow.utils.APIGenerator_request import APIGenerator_request

# 这里或许未来可以有个pipeline基类
class ReasoningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        api_generator = APIGenerator_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                
                max_workers=100
        )

        self.question_filter_step1 = QuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            generator=api_generator
        )
        self.question_gen_step2 =  QuestionGenerator(
            num_prompts=3,
            generator=api_generator
        )
        self.question_filter_step3 = QuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            generator=api_generator
        )
        self.question_difficulty_classifier_step4 = QuestionDifficultyClassifier(
            generator=api_generator
        )
        self.question_category_classifier_step5 = QuestionCategoryClassifier(
            generator=api_generator
        )

        # 未来或许可以维护一个类似nn.sequential的容器，方便添加并实例化多个算子
    def forward(self):
        self.question_filter_step1.run(
            storage=self.storage.step(),
            input_key = "instruction",
        )
        self.question_gen_step2.run(
            storage=self.storage.step(),
            input_key = "instruction",
        )
        self.question_filter_step3.run(
            storage=self.storage.step(),
            input_key = "instruction",
        )
        self.question_difficulty_classifier_step4.run(
            storage=self.storage.step(),
            input_key = "instruction",
            output_key= "question_difficulty"
        )
        self.question_category_classifier_step5.run(
            storage=self.storage.step(),
            input_key = "instruction",
            output_key= "question_category"
        )
        

model = ReasoningPipeline()
model.forward()

