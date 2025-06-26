from dataflow.operators.generate.Reasoning import (
    QuestionCategoryClassifier,
    QuestionDifficultyClassifier,
    QuestionGenerator,
    AnswerGenerator,
)
from dataflow.operators.process.Reasoning import *
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request, LocalModelLLMServing

# 这里或许未来可以有个pipeline基类
class ReasoningPipeline_Pretrain():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )

        self.question_filter_step1 = QuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=llm_serving
        )
        self.question_gen_step2 =  QuestionGenerator(
            num_prompts=3,
            llm_serving=llm_serving
        )
        
        ########################## branch ############################
        self.answer_pipeline_root_step3 = AnswerPipelineRoot()
        ########################## answer ############################
        self.answer_generator_step4 = AnswerGenerator(
            llm_serving=llm_serving
        )
        
        self.answer_ngram_filter_step5 = AnswerNgramFilter(
            min_score = 0.1,
            max_score = 1.0,
            ngrams = 5
        )
        
        self.answer_format_filter_step6 = AnswerFormatterFilter()
                
        # 未来或许可以维护一个类似nn.sequential的容器，方便添加并实例化多个算子
    def forward(self):

        # self.question_filter_step1.run(
        #     storage = self.storage.step(),
        #     input_key = "instruction",
        # )

        # self.question_gen_step2.run(
        #     storage = self.storage.step(),
        #     input_key = "instruction",
        # )

        # ############# branch #############
        # self.answer_pipeline_root_step3.run(
        #     storage = self.storage.step(),
        #     input_answer_key = "output",
        #     input_gt_key = "golden_answer"
        # )
        # ############## answer #############
        # self.answer_generator_step4.run(
        #     storage = self.storage.step(),
        #     input_key = "instruction", 
        #     output_key = "generated_cot"
        # )
        # self.answer_ngram_filter_step5.run(
        #     storage = self.storage.step(),
        #     question_key = "instruction",
        #     answer_key = "generated_cot"
        # )
        self.answer_format_filter_step6.run(
            storage = self.storage.step(),
            read_key_question="question",
            read_key_answer="answer",
            output_key="text",
            )
        
pipeline = ReasoningPipeline_Pretrain()
pipeline.forward()

