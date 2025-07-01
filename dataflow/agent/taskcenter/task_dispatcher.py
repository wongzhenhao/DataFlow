import json
import requests
import yaml
from typing import Dict, Any
import os
from enum import Enum
from ..promptstemplates.prompt_template import PromptsTemplateGenerator
import importlib
import inspect
import pickle
import yaml
from typing import Callable, Dict, Any, Union,List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

FuncSpec = Dict[str, str]
def func2spec(fn: Callable) -> FuncSpec:
    """
    把 *模块级* 函数转成可 JSON/pickle 的描述
    """
    mod = fn.__module__
    qual = fn.__qualname__
    return {"module": mod, "qualname": qual}

def spec2func(spec: FuncSpec) -> Callable:
    """
    通过描述信息重新拿回函数对象
    """
    mod = importlib.import_module(spec["module"])
    fn  = mod
    for attr in spec["qualname"].split("."):
        fn = getattr(fn, attr)
    return fn

def is_module_level_fn(fn: Any) -> bool:
    return inspect.isfunction(fn) and fn.__qualname__ == fn.__name__

class Task:
    def __init__(self, config_path, prompts_template:PromptsTemplateGenerator,
                 system_template: str, task_template: str,
                 param_funcs: dict,is_result_process = False, task_result_processor = None,use_pre_task_result = True,task_name = None):
        # ------------------- 配置 -------------------
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.config_path      = config_path          # 反序列化时要用
        self.api_key          = config["api_key"]
        self.base_url         = config["base_url"]
        self.modelname        = config["modelname"]
        
        # self.tasktype         = TaskType(config["tasktype"])

        # ------------------- 运行期字段 -------------------
        self.prompts_template   = prompts_template
        self.param_funcs        = param_funcs         # 里头有不可 pickle 的
        self.task_result_processor = task_result_processor
        # ------------------- 纯数据字段 -------------------
        self.task_name        = task_name
        self.system_template  = system_template
        self.task_template    = task_template
        self.sys_prompts      = ""
        self.task_prompts     = ""
        self.done             = False
        self.task_result      = ""
        self.pre_task_result: Union[Dict[str, Any], List[Dict[str, Any]]] = {}
        self.use_pre_task_result = use_pre_task_result
        self.depends_on       = [] #任务依赖，默认依赖前一个任务的结果
        self.task_params      = None
        self.is_result_process= is_result_process
        if self.task_name in config:
            self.tool_call_map= config[self.task_name]    
        else:
            self.tool_call_map= {}

    def render_templates(self):
        """
        Dynamically render all templates based on task parameters.
        """
        if not self.task_params or "templates" not in self.task_params:
            print("No templates to render.")
            return
        for template_cfg in self.task_params["templates"]:
            template_name = template_cfg["name"]
            params = template_cfg.get("params", {})
            try:
                rendered = self.prompts_template.render(
                    template_name=template_name,
                    **params
                )
                if "system" in template_name.lower():
                    self.sys_prompts = rendered
                else:
                    self.task_prompts = rendered
            except Exception as e:
                print(f"Error rendering template {template_name}: {str(e)}")


    def __getstate__(self):
        state = self.__dict__.copy()

        state["_prompts_template_cls"] = (
            self.prompts_template.__class__.__module__,
            self.prompts_template.__class__.__qualname__,
        )
        state["prompts_template"] = None
        serializable_funcs = {}
        for name, fn in self.param_funcs.items():
            if is_module_level_fn(fn):
                serializable_funcs[name] = func2spec(fn)
            else:
                # 对于 local_tool_for_sample 这类 bound-method，只保存占位信息
                serializable_funcs[name] = {"bound_sampler": True}
        state["param_funcs"] = serializable_funcs
        trp = self.task_result_processor
        if trp and is_module_level_fn(trp):
            state["task_result_processor"] = func2spec(trp)
            state["_trp_is_spec"] = True
        else:
            state["_trp_is_spec"] = False

        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        # ---- prompts_template ----
        mod_name, qual_name = self.__dict__.pop("_prompts_template_cls")
        mod = importlib.import_module(mod_name)
        cls = mod
        for attr in qual_name.split("."):
            cls = getattr(cls, attr)
        self.prompts_template = cls()       # 重新实例化
        # ---- param_funcs ----
        restored = {}
        for name, spec in self.param_funcs.items():
            if spec == {"bound_sampler": True}:
                # 需要新的 Sampler 实例
                from ..toolkits import local_tool_for_sample
                restored[name] = local_tool_for_sample
            else:
                restored[name] = spec2func(spec)
        self.param_funcs = restored

        # ---- task_result_processor ----
        if self.__dict__.pop("_trp_is_spec"):
            self.task_result_processor = spec2func(self.task_result_processor)