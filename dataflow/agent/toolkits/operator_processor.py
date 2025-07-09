import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from .tools import ChatAgentRequest
from typing import Dict,Any, List
from dataflow import get_logger
logger = get_logger()

def _py_literal(val: Any) -> str:
    if isinstance(val, str):
        return repr(val)         
    if val is None:
        return '""'
    return repr(val)

def _find_first_operator(module) -> type:
    """返回模块中第一个同时满足  (1) 是类  (2) 继承 OperatorABC  (3) 在 OPERATOR_REGISTRY 里  的类"""
    from dataflow.core import OperatorABC
    from dataflow.utils.registry import OPERATOR_REGISTRY

    for obj in module.__dict__.values():
        if inspect.isclass(obj) and issubclass(obj, OperatorABC) and obj is not OperatorABC:
            if OPERATOR_REGISTRY.get(obj.__name__) is obj:
                return obj
    raise RuntimeError("未在该文件中找到符合要求的算子类")

def generate_operator_py(
    request: ChatAgentRequest,
) -> str:
    src_path = Path(request.py_path).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    
    original_code = src_path.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(src_path.stem, src_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[src_path.stem] = module
    spec.loader.exec_module(module)
    operator_cls = _find_first_operator(module)
    cls_name = operator_cls.__name__
    # 解析 __init__，构造实例化形参（带默认值）
    sig = inspect.signature(operator_cls.__init__)

    init_parts = []
    op_need_llm = False
    llm_block = ' '
    for name, param in list(sig.parameters.items())[1:]:
        if name == "llm_serving":
            init_parts.append("llm_serving=llm_serving")
            op_need_llm = True
        elif param.default is param.empty:
            init_parts.append(f"# TODO: {name}=...")
        else:
            init_parts.append(f"{name}={_py_literal(param.default)}")
    init_call = f"{cls_name}({', '.join(init_parts)})"

    # ---------- LLM-Serving 代码块 ----------
    if op_need_llm:
        if request.use_local_model:
            # 暂时不写，等确定的Serving
            llm_block = textwrap.dedent(
                f"""\
                # -------- LLM Serving (Local) --------
                llm_serving = LocalModelLLMServing_vllm(
                    hf_model_name_or_path="{request.local_model_name_or_path}",
                    vllm_tensor_parallel_size=1,
                    vllm_max_tokens=8192,
                    hf_local_dir="local",
                )
                """
            )
                # f"""\
                # # -------- LLM Serving (Local) --------
                # llm_serving = LocalModelLLMServing_vllm(
                #     hf_model_name_or_path="{request.local_model_name_or_path}",
                #     vllm_tensor_parallel_size=1,
                #     vllm_max_tokens=8192,
                #     hf_local_dir="local",
                # )
                # """
        else:
            llm_block = textwrap.dedent(
                f"""\
                # -------- LLM Serving (Remote) --------
                llm_serving = APILLMServing_request(
                    api_url="https://api.openai.com/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=100,
                )
                # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True
                """
            )

    # ---------- main 代码块 ----------
    main_block = textwrap.dedent(
        f"""
        # ======== Auto-generated runner ========
        from dataflow.utils.storage import FileStorage
        from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
        from dataflow.core import LLMServingABC

        if __name__ == "__main__":
            # 1. FileStorage
            storage = FileStorage(
                first_entry_file_name="{request.json_file}",
                cache_path="./cache_local",
                file_name_prefix="dataflow_cache_step",
                cache_type="jsonl",
            )

            # 2. LLM-Serving
        """
    ) + textwrap.indent(llm_block, " " * 4) + textwrap.dedent(
        f"""
            # 3. Instantiate operator
            operator = {init_call}

            # 4. Run
            operator.run(storage=storage.step())
        """
    )

    full_code = f"{original_code.rstrip()}\n\n{main_block}"

    output_path = Path(request.py_path).expanduser().resolve()
    output_path.write_text(full_code, encoding="utf-8")
    return full_code

def local_tool_for_get_match_operator_code(pre_task_result: Dict[str, Any]) -> str:
    """
    According to the names in `pre_task_result["match_operators"]`, look up the corresponding operator classes
    in OPERATOR_REGISTRY, extract their full source code, and concatenate the results.

    If an operator cannot be found or its source code cannot be retrieved, a comment will be added in the result.

    Returns
    -------
    str
        The source code text of all matched operators, separated by two newlines.
    """
    if not pre_task_result or not isinstance(pre_task_result, dict):
        return "# ❗ pre_task_result is empty, cannot extract operator names"
    from dataflow.utils.registry import OPERATOR_REGISTRY
    _NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}
    blocks: List[str] = []
    for op_name in pre_task_result.get("match_operators", []):
        cls = _NAME2CLS.get(op_name)
        if cls is None:
            blocks.append(f"# --- {op_name} is not registered in OPERATOR_REGISTRY ---")
            continue
        try:
            cls_src = inspect.getsource(cls)
            module_src = inspect.getsource(sys.modules[cls.__module__])
            import_lines: list[str] = []
            for line in module_src.splitlines():
                if line.strip().startswith(("import ", "from ")):
                    import_lines.append(line)
                else:
                    if import_lines:
                        break
            import_block = "\n".join(import_lines)
            src_block = f"# === Source of {op_name} ===\n{import_block}\n\n{cls_src}"
            blocks.append(src_block)
        except (OSError, TypeError) as e:
            blocks.append(f"# --- Failed to get the source code of {op_name}: {e} ---")
    return "\n\n".join(blocks)

def local_tool_for_debug_and_exe_operator(
    request: ChatAgentRequest,
    *,
    dry_run: bool = False,
    is_in_debug_process : bool = False,
    current_round: int = 0
):
    py_file = Path(request.py_path)
    if not py_file.exists():
        raise FileNotFoundError(f"Operator file does not exist: {py_file}")
    else:
        code = generate_operator_py(request=request)
    # if current_round == 0 and py_file.exists():
    #     code = generate_operator_py(request=request)
    #     logger.info(f"Reusing existing opetator file {request.py_path}")
    # else:
    #     code = py_file.read_text(encoding="utf-8")
    logger.info(f"[Agent Generated Operator Code in {request.py_path}]:\n{code}")
    if dry_run:
        # Only surface the code to the caller – no execution.
        logger.info("[Dry-run] Operator will not be executed.")
        return code
    if request.execute_the_operator:
        logger.info("\n[............Operator is running............]\n")
        run_res = subprocess.run(
            [sys.executable, str(py_file)],
            capture_output=True,
            text=True,
        )
        if run_res.returncode != 0:
            raise RuntimeError(
                f"{py_file} exited with {run_res.returncode}\n"
                f"stdout:\n{run_res.stdout}\n"
                f"stderr:\n{run_res.stderr}"
            )
        logger.info(
            f"\n[............Operator {request.py_path} executed successfully............]\n"
            f"stdout:\n{run_res.stdout}"
        )
    return code