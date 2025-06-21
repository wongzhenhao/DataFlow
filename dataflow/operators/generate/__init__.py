import sys
from dataflow.utils.registry import LazyLoader

cur_path = "dataflow/operators/generate/"
_import_structure = {
    "AnswerGenerator": (cur_path + "Reasoning/AnswerGenerator.py", "AnswerGenerator"),
    "QuestionCategoryClassifier": (cur_path + "Reasoning/QuestionCategoryClassifier.py", "QuestionCategoryClassifier"),
    "QuestionDifficultyClassifier": (cur_path + "Reasoning/QuestionDifficultyClassifier.py", "QuestionDifficultyClassifier"),
    "QuestionGenerator": (cur_path + "Reasoning/QuestionGenerator.py", "QuestionGenerator"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/generate/", _import_structure)
