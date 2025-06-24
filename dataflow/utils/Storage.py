from abc import ABC, abstractmethod
import pandas as pd
import json
from typing import Any, Literal
import os


class DataFlowStorage(ABC):
    """
    Abstract base class for data storage.
    """
    @abstractmethod
    def read(self, output_type) -> Any:
        """
        Read data from file.
        type: type that you want to read to, such as "datatrame", List[dict], etc.
        """
        pass
    
    @abstractmethod
    def write(self, data: Any) -> Any:
        pass

class FileStorage(DataFlowStorage):
    """
    Storage for file system.
    """
    def __init__(self, 
                 first_entry_file_name: str,
                 cache_path:str="./cache",
                 file_name_prefix:str="dataflow_cache_step",
                 cache_type:Literal["json", "jsonl", "csv", "parquet", "pickle"] = "jsonl"
                 ):
        self.first_entry_file_name = first_entry_file_name
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        self.operator_step = -1

    def _get_cache_file_path(self, step) -> str:
        if step == 0:
            # If it's the first step, use the first entry file name
            return os.path.join(self.first_entry_file_name)
        else:
            return os.path.join(self.cache_path, f"{self.file_name_prefix}_{step}.{self.cache_type}")

    def step(self):
        self.operator_step += 1
        return self
    
    def reset(self):
        self.operator_step = -1
        return self

    def read(self, output_type: Literal["dataframe", "dict"]) -> Any:
        """
        Read data from current file managed by storage.
        output_type: type that you want to read to, such as "dataframe", List[dict], etc.
        """
        file_path = self._get_cache_file_path(self.operator_step)
        print(f"Reading data from {file_path} with type {output_type}")

        if self.operator_step == 0:
            local_cache = file_path.split(".")[-1]
        else:
            local_cache = self.cache_type

        # load data from file
        if local_cache == "json":
            dataframe = pd.read_json(file_path)
        elif local_cache == "jsonl":
            dataframe = pd.read_json(file_path, lines=True)
        elif local_cache == "csv":
            dataframe = pd.read_csv(file_path)
        elif local_cache == "parquet":
            dataframe = pd.read_parquet(file_path)
        elif local_cache == "pickle":
            dataframe = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.cache_type}, input file should end with json, jsonl, csv, parquet, pickle")
        
        if output_type == "dataframe":
            return dataframe
        elif output_type == "dict":
            return dataframe.to_dict(orient="records")
        raise ValueError(f"Unsupported type: {output_type}, type should be dataframe or dict")
        
    def write(self, data: Any) -> Any:
        """
        Write data to current file managed by storage.
        data: Any, the data to write, it should be a dataframe, List[dict], etc.
        """
        if type(data) == list:
            if type(data[0]) == dict:
                dataframe = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data[0])}")
        elif type(data) == pd.DataFrame:
            dataframe = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        file_path = self._get_cache_file_path(self.operator_step + 1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Writing data to {file_path} with type {self.cache_type}")
        if self.cache_type == "json":
            dataframe.to_json(file_path, orient="records", force_ascii=False)
        elif self.cache_type == "jsonl":
            dataframe.to_json(file_path, orient="records", lines=True, force_ascii=False)
        elif self.cache_type == "csv":
            dataframe.to_csv(file_path, index=False)
        elif self.cache_type == "parquet":
            dataframe.to_parquet(file_path)
        elif self.cache_type == "pickle":
            dataframe.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.cache_type}, output file should end with json, jsonl, csv, parquet, pickle")
        
        return file_path