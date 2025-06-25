import re
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class HtmlUrlRemoverRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    
    @staticmethod
    def get_desc(self, lang):
        return "去除文本中的URL和HTML标签" if lang == "zh" else "Remove URLs and HTML tags from the text."

    def run(self, storage: DataFlowStorage, input_keys: list, output_key: str):
        self.input_keys = input_keys
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_keys].to_dict(orient='records'), desc=f"Implementing {self.__class__.__name__}"):
            if isinstance(item, dict):
                modified = False
                for key in input_keys:
                    if key in item and isinstance(item[key], str):
                        original_text = item[key]
                        refined_text = original_text

                        # Remove URLs
                        refined_text = re.sub(r'https?:\/\/\S+[\r\n]*', '', refined_text, flags=re.MULTILINE)
                        # Remove HTML tags
                        refined_text = re.sub(r'<.*?>', '', refined_text)

                        if original_text != refined_text:
                            item[key] = refined_text
                            modified = True
                            self.logger.debug(f"Modified text for key '{key}': Original: {original_text[:30]}... -> Refined: {refined_text[:30]}...")

                refined_data.append(item)
                if modified:
                    numbers += 1
                    self.logger.debug(f"Item modified, total modified so far: {numbers}")

        dataframe[self.output_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.output_key]