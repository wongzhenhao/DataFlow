import inspect
import os
from pathlib import Path
import subprocess
import sys
from .tools import ChatAgentRequest
from typing import Dict,Any, List
from dataflow import get_logger
logger = get_logger()

def generate_operator_py(request:ChatAgentRequest,special_model:str):
    pass

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
):
    py_file = Path(request.py_path)
    if not py_file.exists():
        raise FileNotFoundError(f"Operator file does not exist: {py_file}")
    # Always read source code for logging / return
    code = py_file.read_text(encoding="utf-8")
    logger.info(f"[Agent Generated Operator Code in {request.py_path}]:\n{code}")
    if dry_run:
        # Only surface the code to the caller – no execution.
        logger.info("[Dry-run] Operator will not be executed.")
        return code
    if getattr(request, "execute_the_operator", True):
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