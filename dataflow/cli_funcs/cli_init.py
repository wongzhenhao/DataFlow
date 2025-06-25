import os
import re
from colorama import init, Fore, Style
from .paths import DataFlowPath
from .copy_funcs import copy_files_without_recursion, copy_file, copy_files_recursively

def _copy_scripts():
    current_dir = os.getcwd()
    # Copy train scripts
    target_dir = os.path.join(current_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # script_path = DataFlowPath.get_dataflow_scripts_dir()

    copy_files_recursively(DataFlowPath.get_dataflow_scripts_dir(), target_dir)

    
def cli_init(subcommand):
    print(f'{Fore.GREEN}Initializing in current working directory...{Style.RESET_ALL}')
    
    # base initialize that only contain default scripts
    if subcommand == "base":
        _copy_scripts()
        
    # if subcommand == "model_zoo":
    #     _copy_train_scripts()
    #     _copy_demo_runs() 
    #     _copy_demo_configs()
    #     _copy_dataset_json()
    # # base initialize that only contain default scripts
    # if subcommand == "backbone":
    #     _copy_train_scripts()
    #     _copy_demo_runs() 
    #     _copy_demo_configs()
    #     _copy_dataset_json()
    # print(f'{Fore.GREEN}Successfully initialized IMDLBenCo scripts.{Style.RESET_ALL}')