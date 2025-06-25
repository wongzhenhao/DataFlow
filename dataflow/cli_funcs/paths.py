import os
from pathlib import Path
import appdirs


class DataFlowPath:
    """
    Class to manage paths for DataFlow.
    """
    
    @staticmethod
    def get_dataflow_dir() -> Path:
        # return path of /dataflow
        return Path(__file__).parent.parent 

    @staticmethod
    def get_dataflow_scripts_dir() -> Path:
        return DataFlowPath.get_dataflow_dir() / "scripts"

    # @staticmethod
    # def get_dataset_json_dir() -> Path:
    #     return DataFlowPath.get_dataflow_dir() / "dataset_json"

    # @staticmethod
    # def get_init_base_dir() -> Path:
    #     return DataFlowPath.get_dataflow_dir() / "init_base"

    # @staticmethod
    # def get_model_zoo_runs_dir() -> Path:
    #     return DataFlowPath.get_dataflow_dir() / "model_zoo" / "runs"