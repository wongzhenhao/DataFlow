# 根节点，用来将数据拆入不同的分支
import pandas as pd
# from dataflow.generator.algorithms.AnswerExtraction_qwenmatheval import UnitTextManager,StringCleaner,AnswerExtractor
from dataflow.utils.utils import get_generator
import logging
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger
import re
from word2number import w2n
import pandas as pd
from tqdm import tqdm
from dataflow.utils import Operator

class StringProcessor:
    """
    A class that encapsulates various string processing functions for mathematical expressions.
    """

    @staticmethod
    def _fix_fracs(string):
        """
        Fixes fraction expressions in the string, ensuring they are properly formatted as \frac{a}{b}.
        """
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            for substr in substrs[1:]:
                new_str += "\\frac"
                if len(substr) > 0 and substr[0] == "{":
                    new_str += substr
                else:
                    if len(substr) >= 2:
                        a, b = substr[0], substr[1]
                        if b != "{":
                            new_str += f"{{{a}}}{{{b}}}{substr[2:]}" if len(substr) > 2 else f"{{{a}}}{{{b}}}"
                        else:
                            new_str += f"{{{a}}}{b}{substr[2:]}" if len(substr) > 2 else f"{{{a}}}{b}"
                    else:
                        return string
        return new_str

    @staticmethod
    def _fix_a_slash_b(string):
        """
        Fixes cases where a fraction is represented as a simple division (e.g., a/b) and converts it to \frac{a}{b}.
        """
        if len(string.split("/")) != 2:
            return string
        a, b = string.split("/")
        try:
            a, b = int(a) if "sqrt" not in a else a, int(b) if "sqrt" not in b else b
            assert string == f"{a}/{b}"
            return f"\\frac{{{a}}}{{{b}}}"
        except:
            return string

    @staticmethod
    def _fix_sqrt(string):
        """
        Ensures that square root expressions are properly formatted as \sqrt{...}.
        """
        return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)

    @staticmethod
    def convert_word_number(text: str) -> str:
        """
        Converts a word representation of a number to a digit.
        """
        try:
            return str(w2n.word_to_num(text))
        except:
            return text


# Unit Text Class to Manage Unit Texts
class UnitTextManager:
    """
    A class that encapsulates unit text management to remove unwanted unit terms from strings.
    """

    def __init__(self):
        """
        Initializes the unit texts and their plural forms.
        """
        self.unit_texts = [
            "east", "degree", "mph", "kmph", "ft", "m sqaure", "m east", "sq m", "deg", "mile", "q .", "monkey", "prime",
            "ratio", "profit of rs", "rd", "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west",
            "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h", "no change", "men", "soldier", "pie", "bc",
            "excess", "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise", "a . m", "th", "π r 2", "sq",
            "mark", "l", "toy", "coin", "sq . m", "gallon", "° f", "profit", "minw", "yr", "women", "feet", "am", "pm", "hr",
            "cu cm", "square", "v â € ™", "are", "rupee", "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day",
            "hour", "minute", "min", "second", "man", "woman", "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³", "hectare",
            "more", "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm", "apple", "liter",
            "loss", "yard", "pure", "year", "increase", "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .", "metre",
            "meter", "inch",
        ]
        self.unit_texts.extend([t + "s" for t in self.unit_texts])

    def clean_units(self, string: str):
        """
        Cleans the string by removing unit terms from it.
        """
        for unit_text in self.unit_texts:
            string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
        return string


# Main String Processing Class
class StringCleaner:
    """
    A class responsible for cleaning and formatting strings in mathematical expressions.
    """

    def __init__(self, unit_manager: UnitTextManager):
        """
        Initializes the StringCleaner class with a unit manager.
        """
        self.unit_manager = unit_manager

    def strip_string(self, string, skip_unit=False):
        """
        Strips unwanted characters and units from the string.
        """
        string = str(string).strip().replace("\n", "").rstrip(".").replace("\\!", "")
        string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
        string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string).replace("bmatrix", "pmatrix")
        string = string.replace("tfrac", "frac").replace("dfrac", "frac").replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
        string = string.replace("\\left", "").replace("\\right", "").replace("\\{", "{").replace("\\}", "}")
        
        # Clean unit texts if needed
        if not skip_unit:
            string = self.unit_manager.clean_units(string)

        string = string.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "").replace("\\(", "").replace("\\)", "")
        string = StringProcessor.convert_word_number(string)
        string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
        
        for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
            string = string.replace(key, "")
        
        string = string.replace("\\emptyset", r"{}").replace("(-\\infty,\\infty)", "\\mathbb{R}")
        string = string.replace("%", "").replace(" .", " 0.").replace("{.", "{0.")
        
        return string


# Core Answer Extraction Logic Class
class AnswerExtractor:
    """
    A class responsible for extracting the final answer from a prediction string.
    """

    def __init__(self, string_cleaner: StringCleaner):
        """
        Initializes the AnswerExtractor class with a string cleaner.
        """
        self.string_cleaner = string_cleaner
        self.logger = get_logger()

    def extract_answer(self, pred_str, data_name, use_last_number=True):
        """
        Extracts the final answer from the prediction string, processing various formats.
        """
        if not pred_str:
            pred_str = ""
        pred_str = pred_str.replace("\u043a\u0438", "")
        
        # Handle special cases based on data_name or pattern
        if "final answer is $" in pred_str and "$. I hope" in pred_str:
            pred = pred_str.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
        elif "boxed" in pred_str:
            pred = self._extract_boxed_answer(pred_str)
        elif "he answer is" in pred_str:
            pred = pred_str.split("he answer is")[-1].strip()
        else:
            pred = self._get_last_number_answer(pred_str, use_last_number)
        
        pred = self.string_cleaner.strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
        return pred

    def _extract_boxed_answer(self, pred_str):
        """
        Extracts answers enclosed in 'boxed' notation.
        """
        ans = pred_str.split("boxed")[-1]
        if ans.startswith("{"):
            return self._extract_bracketed_answer(ans)
        else:
            return ans.split("$")[0].strip()

    def _extract_bracketed_answer(self, ans):
        """
        Handles answers that are enclosed within brackets.
        """
        stack = 1
        result = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                result += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                result += c
            else:
                result += c
        return result

    def _get_last_number_answer(self, pred_str, use_last_number):
        """
        Extracts the last number from the string if use_last_number is True.
        """
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            return pred[-1] if pred else ""
        return ""




@GENERATOR_REGISTRY.register()
class AnswerPipelineRoot(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.input_file = config.get("input_file")
        self.output_file_with_gt = config.get("output_file_with_gt")
        self.output_file_without_gt = config.get("output_file_without_gt")
        self.input_key = config.get("input_key")
        self.input_answer_key = config.get("input_answer_key")
        self.input_gt_key = config.get("input_gt_key", "")
        self.logger = get_logger()

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'generator_type']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "答案处理流程根节点，负责将输入数据根据有无真实标签GT分发到不同处理分支。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- output_dir：输出目录路径\n"
                "- branch_config：分支配置参数\n"
                "- parallel_workers：并行工作线程数\n\n"
                "输出参数：\n"
                "- 多个输出文件路径（根据分支配置生成）"
            )
        elif lang == "en":
            return (
                "Root node of answer processing pipeline, distributes input data to different processing branches.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- output_dir: Output directory path\n"
                "- branch_config: Branch configuration parameters\n"
                "- parallel_workers: Number of parallel workers\n\n"
                "Output Parameters:\n"
                "- Multiple output file paths (generated based on branch config)"
            )
        else:
            return "AnswerPipelineRoot routes data to different processing branches."
        
    def _load_input(self):
        return pd.read_json(self.input_file, lines=True)

    def _write_output(self, save_path, dataframe, extractions):
        dataframe.to_json(save_path, orient="records", lines=True)

    def run(self):
        # 读取输入文件
        # df = pd.read_json(self.input_file, lines=True)
        df = self._load_input()
        
        if not self.input_gt_key or self.input_gt_key not in df.columns:
            self.logger.warning("No valid gt key in input file, copy input file to output file without gt")
            return
            
        # 初始化答案提取器
        if self.input_answer_key in df.columns:
            unit_text_manager = UnitTextManager()
            string_cleaner = StringCleaner(unit_text_manager)
            answer_extractor = AnswerExtractor(string_cleaner)
        
            def extract_gt(answer, gt):
                try:
                    if gt != "" and not pd.isna(gt):
                        return gt
                    else:
                        if pd.isna(answer) or answer == "":
                            return None
                        else:
                            return answer_extractor.extract_answer(answer,None,True)
                except Exception as e:
                    self.logger.error(f"Error in extract_gt: {e}", exc_info=True)
                    return None
            
            # 使用 apply 遍历 DataFrame, 避免显式循环索引问题
            df[self.input_gt_key] = df.apply(lambda row: extract_gt(row[self.input_answer_key],
                                                                    row[self.input_gt_key]),
                                            axis=1)
        
        # 拆分有gt和无gt的 DataFrame
        df_with_gt = df[(df[self.input_gt_key].notna()) & (df[self.input_gt_key] != "")]
        df_without_gt = df[(df[self.input_gt_key].isna()) | (df[self.input_gt_key] == "")].copy()
        df_without_gt[self.input_gt_key] = None
        # 输出结果
        # df_with_gt.to_json(self.output_file_with_gt, orient="records", lines=True)
        # df_without_gt.to_json(self.output_file_without_gt, orient="records", lines=True)
        if len(df_with_gt) > 0:
            self._write_output(self.output_file_with_gt, df_with_gt,1)
        if len(df_without_gt) > 0:
            self._write_output(self.output_file_without_gt, df_without_gt,0)
        self.logger.info(f"output {df_with_gt.shape[0]} rows with gt to {self.output_file_with_gt}")
        self.logger.info(f"output {df_without_gt.shape[0]} rows without gt to {self.output_file_without_gt}")
                    
            



        
        