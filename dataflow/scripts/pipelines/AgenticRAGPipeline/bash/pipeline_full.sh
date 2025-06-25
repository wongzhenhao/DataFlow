# # ------------------------------ Question ------------------------------#
# # Step 0, Initial Clustering and filter
echo -e "\033[32m===== [Step 0] Filter  =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/process/MathProblemFilter.yaml --step_name QuestionFilter

# Step 1, Question Synthesis
echo  -e "\033[32m===== [Step 1] Question Synthesis =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/generate/QuestionGenerator.yaml --step_name QuestionGenerator 

# Step 2, Question Correctness Filter
echo -e "\033[32m===== [Step 2] Question Correctness Filter =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/process/MathProblemFilter_step2.yaml --step_name QuestionFilter

# Step 3, Difficulty classification
echo -e "\033[32m===== [Step 3] Difficulty Classification =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/generate/QuestionDifficultyClassifier.yaml --step_name QuestionDifficultyClassifier