import inspect
from typing import List, Dict, Any, Type, Iterable, Tuple
from dataflow.utils.registry import OPERATOR_REGISTRY

def _parse_params(sig: inspect.Signature) -> List[Dict[str, Any]]:
    """把签名对象拆成「参数字典列表」, 并忽略 self。"""
    return [
        {
            "name": p.name,
            "default": None if p.default is inspect.Parameter.empty else p.default,
            "kind": str(p.kind),                 # POSITIONAL_ONLY / VAR_POSITIONAL …
        }
        for p in sig.parameters.values()
        if p.name != "self"
    ]

def get_class_method_params(
    cls: Type,
    method_name: str = "run",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    获取类 `__init__` 以及指定方法的参数信息。

    返回格式:
        {
          "init": [ {...}, ... ],
          "run":  [ {...}, ... ]   # 如果类没有该方法则值为 None
        }
    """
    params: Dict[str, List[Dict[str, Any]] | None] = {}

    # __init__
    params["init"] = _parse_params(inspect.signature(cls.__init__))

    # 指定方法
    if hasattr(cls, method_name):
        params[method_name] = _parse_params(
            inspect.signature(getattr(cls, method_name))
        )
    else:
        params[method_name] = None

    return params


def collect_operator_params(
    method_name: str = "run",
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    遍历 OPERATOR_REGISTRY，收集所有算子的参数信息。

    :param operator_registry: 形如 [(name, cls), ...] 的可迭代对象
    :param method_name:      需要提取的业务方法名，默认 'run'
    :return: {
               "OperatorName": {
                   "init": [...],
                   "run":  [...]
               },
               ...
             }
    """
    summary: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name, cls in OPERATOR_REGISTRY:
        summary[name] = get_class_method_params(cls, method_name=method_name)
    return summary