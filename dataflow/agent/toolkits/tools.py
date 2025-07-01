from __future__ import annotations
import sys
import os
from pydantic import BaseModel
import httpx
import json
import uuid
from typing import List, Dict, Sequence, Any, Union, Optional, Iterable, Mapping, Set, Callable
from pathlib import Path
from termcolor import colored
from functools import lru_cache
import yaml
from clickhouse_connect import get_client
import subprocess
from collections import defaultdict, deque
from dataflow.utils.storage import FileStorage
from .logger import get_logger
logger = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MAX_JSONL_LINES = 50
DATA_DIR = Path("./data/knowledgebase")  # Local data storage directory

class HistoryItem(BaseModel):
    timestamp: Optional[str] = None
    operation: Optional[str] = None
    result: Optional[str] = None

class GlobalVariables(BaseModel):
    dataset_id: str
    feedback: Optional[str] = None
    history: List[HistoryItem] = []


class ChatAgentRequest(BaseModel):
    user_id: Optional[int] = None
    target: str = ""
    model: Optional[str] = None
    kb_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    language: Optional[str] = None
    sessionKEY: str

class ChatResponse(BaseModel):
    id: str
    name: str
    info: Any

def validate_filename(kb_id: str) -> Path:
    """
    Validate and generate a safe file path for the given knowledge base ID.
    """
    if not kb_id.endswith(".jsonl"):
        kb_id += ".jsonl"
    if '/' in kb_id or '\\' in kb_id:
        raise ValueError("Invalid file name")
    file_path = DATA_DIR / kb_id
    if not file_path.exists():
        raise FileNotFoundError(f"File {kb_id} not found")
    return file_path

def parse_jsonl(file_path: Path) -> List[Dict]:
    """
    Parse JSONL data from a local file.
    """
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:MAX_JSONL_LINES]
    except Exception as e:
        raise IOError(f"File read error: {str(e)}")

    for idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"Line {idx} is not a JSON object")
            results.append(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {idx}: {str(e)}")

    return results

def format_context(data: List[Dict]) -> str:
    """
    Format JSON data as natural language context.
    """
    context = ["The following are structured data entries from the knowledge base:"]
    for item in data:
        context.append("Entry contains the following fields:")
        for key, value in item.items():
            context.append(f"- {key}: {str(value)[:200]}")
    return "\n".join(context)

def local_tool_for_get_workflow_bg(request: ChatAgentRequest) -> str:
    """
    Get the workflow background as a JSON string.
    """
    if not request.workflow:
        return ""
    return json.dumps(request.workflow, ensure_ascii=False, indent=2)

def local_tool_for_get_purpose(memory, request:ChatAgentRequest) -> str:
    """
    Get the workflow background as a JSON string.
    """
    # if pre_task_result.get("purpose"):
    #     return pre_task_result.get("purpose")
    # else:
    #     return ""
    purpose = memory.get_session_data(session_id=memory.get_session_id(request.sessionKEY),key = "conversation_router").get('purpose')

    if purpose:
        return purpose
    else:
        return ""



def get_kb_content(request: ChatAgentRequest) -> Optional[Any]:
    """
    Load knowledge base content from a JSON file based on the request.
    """
    kb_id = request.kb_id
    try:
        with open(f"{parent_dir}/data/knowledgebase/{kb_id}.json", 'r', encoding='utf-8') as f:
            kb_content = json.load(f)
        return kb_content
    except FileNotFoundError:
        print(f"Error: The JSON file {kb_id}.json was not found!")
    except json.JSONDecodeError:
        print(f"Error: Unable to parse the JSON file {kb_id}.json!")
    except Exception as e:
        print(f"Error: An unknown error occurred: {e}")
    return None

def get_operator_content(request: Any, data_key: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
    """
    Retrieve operator content from the configuration file based on the data key.
    """
    if isinstance(data_key, list):
        if not data_key:
            print("Error: data_key list is empty!")
            return None
        key_dict = data_key[0]
        if not isinstance(key_dict, dict):
            print(f"Error: The first element of data_key is not a dict, but {type(key_dict)}")
            return None
    elif isinstance(data_key, dict):
        key_dict = data_key
    else:
        print(f"Error: Unsupported data_key type {type(data_key)}; only dict or list[dict] is accepted")
        return None
    try:
        subtype = next(iter(key_dict.values()))
    except StopIteration:
        print("Error: No key-value pairs in data_key!")
        return None
    file_path = "/mnt/h_h_public/lh/lz/DataFlow-AgentBot/Config/Operator.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            operator_content = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file {file_path}!")
        return None
    result = operator_content.get(subtype)
    if result is None:
        print(f"Warning: Key '{subtype}' not found in Operator.json; available keys: {list(operator_content.keys())}")
    return result

def get_operator_content_map_from_all_operators(
    request: Any,
    pre_task_result: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Any:
    """
    Retrieve operator mapping from all operators based on the data key.

    Parameters
    ----------
    request : Any
        Request object (not used in this function, reserved for future use).
    pre_task_result : dict or list of dict
        Data key(s) used to determine operator type.

    Returns
    -------
    Any
        Operator mapping content or None if not found or error.
    """
    data_key = pre_task_result
    if isinstance(data_key, list):
        if not data_key:
            print("Error: data_key list is empty!")
            return None
        key_dict = data_key[0]
        if not isinstance(key_dict, dict):
            print(f"Error: The first element of data_key is not a dict, but {type(key_dict)}")
            return None
    elif isinstance(data_key, dict):
        key_dict = data_key
    else:
        print(f"Error: Unsupported data_key type {type(data_key)}; only dict or list[dict] is accepted")
        return None
    try:
        subtype = next(iter(key_dict.values()))
    except StopIteration:
        print("Error: No key-value pairs in data_key!")
        return None
    if subtype == "MIXTURE":
        result = get_operator_descriptions("dataflow")
        return result
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "resources", "Operator.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            operator_content = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file {file_path}!")
        return None
    result = operator_content.get(subtype)
    if result is None:
        print(f"Warning: Key '{subtype}' not found in Operator.json; available keys: {list(operator_content.keys())}")
    return result

def get_operator_descriptions(root_package: str = "dataflow", lang: str = "zh") -> List[Dict[str, Any]]:
    """
    Scan all classes with a get_desc method under the root package,
    returning a list of dicts with id, name, and description.
    """
    import importlib
    import os
    import inspect
    operator_classes = []

    root_dir = os.path.dirname(importlib.import_module(root_package).__file__)
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                relative_path = os.path.relpath(root, root_dir)
                if relative_path == '.':
                    module_name = f"{root_package}.{file[:-3]}"
                else:
                    module_name = f"{root_package}.{relative_path.replace(os.sep, '.')}.{file[:-3]}"
                module_name = module_name.replace('..', '.')
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and hasattr(obj, 'get_desc'):
                            operator_classes.append(obj)
                except ImportError:
                    continue

    desc_list = []
    for idx, operator in enumerate(operator_classes, 1):
        try:
            if isinstance(operator.__dict__.get('get_desc', None), staticmethod):
                desc = operator.get_desc(lang)
            else:
                desc = operator().get_desc(lang)
        except Exception as e:
            desc = f"Error getting description: {e}"
        desc_list.append({
            "node": idx,
            "name": operator.__name__,
            "description": desc
        })
    return desc_list

def local_tool_for_get_chat_history(
    request,
    memory
):
    session_id = memory.get_session_id(request.sessionKEY)
    history = memory.get_history(session_id)
    return history

def local_tool_for_get_chat_target(
    request:ChatAgentRequest
):
    return request.target

def local_tool_for_get_categories():
    filepath = "/mnt/h_h_public/lh/lz/new-dataagent/toolkits/resources/Operator.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return ','.join(data.keys())

@lru_cache
def local_tool_for_load_yaml_cfg() -> dict:
    """
    读取并缓存配置文件，避免每次请求都 IO。
    """
    cfg_path = Path(__file__).resolve().parent.parent / "ChatAgentYaml.yaml"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(f"配置文件不存在: {cfg_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"解析 YAML 失败: {e}")

from pathlib import Path
from typing import List, Dict, Any
import json

def local_tool_for_iter_json_items(json_file: Path) -> List[Dict[str, Any]]:
    """
    兼容三种格式：
      1. 多行 JSONL（每行一个 JSON 对象）
      2. 单个 JSON 对象
      3. 单行或多行 JSON 数组（[ {...}, {...} ])
    """
    with json_file.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char in "{[":
            try:
                content = f.read()
                obj = json.loads(content)
                if isinstance(obj, dict):
                    return [obj]
                elif isinstance(obj, list):
                    if all(isinstance(item, dict) for item in obj):
                        return obj
                    else:
                        raise ValueError("JSON 数组元素不是对象")
            except json.JSONDecodeError:
                f.seek(0)

        items: List[Dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("JSONL 中某行不是对象")
                items.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"无法解析 JSONL 行：{line}\n{e}") from None
        return items

def local_tool_for_create_table_if_absent(
    cfg: Dict[str, Any],
    schema: List[Dict[str, str]],
    engine_stmt: str = "ENGINE = MergeTree ORDER BY tuple()",
) -> None:
    """
    使用 get_client(...) 与 MyScale/ClickHouse 建立连接，并建表（存在则跳过）。

    Parameters
    ----------
    cfg     :  YAML 中的 MYSCALE 节点（host/port/username/...）
    schema  :  [{"name": "...", "type": "..."}, ...]
    engine_stmt : 完整的 ENGINE 子句
    """
    field_defs = ", ".join(f"{col['name']} {col['type']}" for col in schema)
    ddl = (
        f"CREATE TABLE IF NOT EXISTS {cfg['database']}.{cfg['table']} "
        f"({field_defs}) "
        f"{engine_stmt}"
    )

    client = get_client(
        host=cfg["host"],
        port=cfg["port"],
        username=cfg["username"],
        password=cfg["password"],
        database=cfg["database"],
    )
    try:
        client.command(ddl)
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

def local_tool_for_clean_json(
    data: Any,
    allowed_keys: Sequence[str],
    *,
    concat_keys: Iterable[str] | None = None,
    hoist_keys: Iterable[str] | None = None,
    hoist_children_spec: Dict[str, Sequence[str] | None] | None = None,
    sep: str = " "
) -> Any:
    """
    Recursively clean JSON with support for:
    1. concat_keys: Concatenate all leaf texts under the key into a string.
    2. hoist_keys: Remove the key and promote its value to the parent node.
    3. hoist_children_spec: Retain the key, but remove its child keys and promote their values.
    """
    allowed: Set[str] = set(allowed_keys)
    concat:  Set[str] = set(concat_keys or ())
    hoist:   Set[str] = set(hoist_keys or ())
    hoist_children_spec = hoist_children_spec or {}

    def _flatten_text(node: Any) -> List[str]:
        if node is None:
            return []
        if isinstance(node, (str, int, float, bool)):
            return [str(node)]
        if isinstance(node, Mapping):
            out: List[str] = []
            for v in node.values():
                out.extend(_flatten_text(v))
            return out
        if isinstance(node, list):
            out: List[str] = []
            for item in node:
                out.extend(_flatten_text(item))
            return out
        return [str(node)]

    def _merge_vals(a: Any, b: Any) -> Any:
        if a is None:
            return b
        if b is None:
            return a
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            merged = dict(a)
            merged.update(b)
            return merged
        if isinstance(a, list) and isinstance(b, list):
            return a + b
        return [a, b]

    def _clean(node: Any) -> Any:
        if isinstance(node, list):
            return [_clean(item) for item in node]

        if isinstance(node, Mapping):
            cleaned: dict[str, Any] = {}
            hoisted_up: List[Any] = []

            for k, v in node.items():
                # concat_keys
                if k in concat:
                    concat_val = sep.join(_flatten_text(v))
                    if k in hoist:
                        hoisted_up.append(concat_val)
                    else:
                        cleaned[k] = concat_val
                    continue

                # hoist_keys
                if k in hoist:
                    hoisted_up.append(_clean(v))
                    continue

                # hoist_children_spec
                if k in hoist_children_spec:
                    sub_allowed = hoist_children_spec[k] or []
                    merged_val: Any = None
                    if isinstance(v, Mapping):
                        for sub_k, sub_v in v.items():
                            if sub_allowed and sub_k not in sub_allowed:
                                continue
                            merged_val = _merge_vals(merged_val, _clean(sub_v))
                    else:
                        merged_val = _clean(v)
                    cleaned[k] = merged_val
                    continue

                if k in allowed:
                    cleaned[k] = _clean(v)

            if hoisted_up:
                merged: Any = cleaned if cleaned else None
                for item in hoisted_up:
                    merged = _merge_vals(merged, item)
                return merged

            return cleaned

        return node

    return _clean(data)

import subprocess
import re
from pathlib import Path
from typing import List

def local_tool_for_mineru_extrac_pdfs(
    p: str,
    o: str,
    d: str = "cpu",
    source: str = "modelscope",
    *,
    strict_error_check: bool = True,
) -> List[str]:
    """
    Wrapper for the CLI tool **mineru**.

    Parameters
    ----------
    p : str
        Input directory, passed to the `-p` flag of `mineru`.
    o : str
        Output directory, passed to the `-o` flag of `mineru`.
    source : str, default ``"modelscope"``
        Source framework, passed to `--source`.
    strict_error_check : bool, default ``True``
        If ``True``, *any* occurrence of the substring ``"error"`` (case-insensitive)
        in the combined stdout / stderr will be treated as a failure,
        even when the process exits with code 0.

    Returns
    -------
    List[str]
        Absolute paths of every JSON file inside *o* (recursively)
        whose filename contains the substring ``"content_list"`` (case-insensitive).

    Raises
    ------
    RuntimeError
        If the subprocess exits with non-zero status,
        if an ``error`` is detected in the output,
        or if no matching JSON files are found.
    FileNotFoundError
        If the `mineru` executable cannot be located.

    """

    # ---------- 1. Build and launch subprocess ---------------------------------
    cmd = ["mineru", "-p", p, "-o", o, "-d", d, "--source", source]

    # text=True → returns str instead of bytes in stdout/stderr
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = (proc.stdout or "") + (proc.stderr or "")

    # ---------- 2. Basic health checks ----------------------------------------
    if proc.returncode != 0:
        raise RuntimeError(
            f"`mineru` exited with code {proc.returncode}.\n"
            f"--- STDOUT ---\n{proc.stdout}\n"
            f"--- STDERR ---\n{proc.stderr}"
        )

    if strict_error_check and re.search(r"\berror\b", combined_output, flags=re.I):
        # False-positive filtering can be inserted here if necessary
        raise RuntimeError(
            "The word 'error' was detected in mineru output although exit code is 0.\n"
            f"Full output:\n{combined_output}"
        )

    # ---------- 3. Locate expected output files ------------------------------
    out_dir = Path(o).expanduser().resolve()
    if not out_dir.is_dir():
        raise RuntimeError(f"Output directory does not exist: {out_dir}")

    json_paths = sorted(
        [str(pth) for pth in out_dir.rglob("*content_list*.json")]
    )

    if not json_paths:
        raise RuntimeError(
            "Execution succeeded but no '*content_list*.json' files were found "
            f"under {out_dir}"
        )

    # ---------- 4. Return results --------------------------------------------
    return json_paths

def _topological_sort(nodes: List[Dict[str, Any]],
                      edges: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Kahn's algorithm – return the topological order of nodes.
    Raise ValueError if the DAG is invalid (contains cycles).
    """
    id2node = {n["id"]: n for n in nodes}
    indeg = defaultdict(int)
    graph = defaultdict(list)

    for e in edges:
        src, dst = e["source"], e["target"]
        graph[src].append(dst)
        indeg[dst] += 1
        # Ensure all nodes appear in indeg
        indeg[src] += 0

    # Also handle isolated nodes (no in-degree and no out-degree)
    for n in id2node:
        indeg[n] += 0

    q = deque([nid for nid, d in indeg.items() if d == 0])
    order: List[str] = []
    while q:
        nid = q.popleft()
        order.append(nid)
        for nxt in graph[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(order) != len(id2node):
        raise ValueError("Graph has a cycle or missing edges; "
                         f"sorted {len(order)} / {len(id2node)} nodes.")

    return [id2node[nid] for nid in order]

# 412 Version !
def local_tool_for_execute_the_recommended_pipeline(
    pre_task_result: Dict[str, Any],
    memory,
    request: ChatAgentRequest,
    *,
    sh_path: str = "run_pipeline.sh",
    env_vars: Dict[str, str] | None = None,
    execute: bool = True,
    dry_run: bool = False,
    log_key: str = "pipeline_log"
) -> str:
    """
    Main entry point.

    Parameters
    ----------
    pre_task_result : dict          # Must follow the specified format
    sh_path         : str           # File path to save the generated shell script
    env_vars        : dict|None     # Environment variables to be exported
    execute         : bool          # Whether to execute the script after generation
    dry_run         : bool          # If True, do not write or execute, just return script content
    log_key         : str 

    Returns
    -------
    sh_content : str  The complete generated shell script content
    """
    ctx = pre_task_result
    nodes, edges = ctx["nodes"], ctx.get("edges", [])
    ordered_nodes = _topological_sort(nodes, edges)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines: list[str] = ["#!/usr/bin/env bash", "set -e", ""]
    lines.append(f"cd {base_dir}/DataFlow-421")
    lines.append(f"export USE_DB=False")

    if env_vars:
        lines.append("# -------- env -------- #")
        for k, v in env_vars.items():
            lines.append(f'export {k}="{v}"')
        lines.append("")

    lines.append("# -------- pipeline -------- #")

    pipeline_q_dir = f"{base_dir}/DataFlow-421/demos/demos_result/math/pipeline_Q"
    step0_file = os.path.join(pipeline_q_dir, "Corrected_step0.jsonl")
    step_offset = 0
    if not os.path.exists(step0_file):
        lines.append('echo -e "\\033[32m===== [Step 0] Initial Filter =====\\033[0m"')
        lines.append(
            "python process.py --config "
            "configs/process/math/pipeline_Q/test_process_math.yaml"
        )
        lines.append("")
        step_offset = 1

    for step, node in enumerate(ordered_nodes, start=step_offset):
        title = node.get("name", node["id"])
        lines.append(f'echo -e "\\033[32m===== [Step {step}] {title} =====\\033[0m"')
        lines.append(node["command"])
        # memory.append_session_list(memory.get_session_id(request.sessionKEY), log_key, node["command"])
        lines.append("")

    sh_content = "\n".join(lines)

    # 只是把sh内容写进去呢
    

    if dry_run:
        return sh_content

    with open(sh_path, "w", encoding="utf-8") as f:
        f.write(sh_content)
    os.chmod(sh_path, 0o755)

    if execute:
        logger.info(f"[INFO] Running pipeline: bash {sh_path}")
        stream_and_capture(
            sh_path     = sh_path,
            memory      = memory,
            session_id  = memory.get_session_id(request.sessionKEY),
            log_path    = "pipeline.log",
            key         = log_key,
        )
    return sh_content

def stream_and_capture(
    sh_path: str,
    memory,
    session_id: str,
    *,
    log_path: str = "pipeline.log",
    key: str = "pipeline_log",
):
    proc = subprocess.Popen(
        ["bash", sh_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:           
        sys.stdout.write(line)
        memory.append_session_list(session_id, key, line.rstrip("\n"))
    retcode = proc.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, proc.args)
    memory.set_session_data(session_id, f"{key}:done", True)

def main_print_logo():
    from pyfiglet import Figlet
    from rich.console import Console
    console = Console()
    fig = Figlet(font="block")
    ascii_art = fig.renderText("DataFlow-Agent").splitlines()
    layers = [
        (0, 0, "red"),    
    ]
    for dx, dy, color in layers:
        if dy:
            console.print("\n" * dy, end="")
        pad = " " * dx
        for line in ascii_art:
            console.print(pad + line, style=color, highlight=False, overflow="ignore")

def local_tool_for_sample(
    sample_size: int = 100,
    use_file_sys: int = 1,
    json_file: str = "",
    cache_type: str = "jsonl"
) -> Dict[str, Any]:
    from collections import Counter
    """
    Sample, classify, and compute statistics on sample data.

    Args:
        storage: Optional storage backend. If None, use default JSONAnyStorage(json_file).
        sample_size: Number of samples to retrieve.
        json_file: Path for JSON storage if storage is not provided.

    Returns:
        A dictionary with overall statistics and sample details.
    """
    def judge_type(sample: Dict[str, Any]) -> str:
        """
        Judge and return the type of a sample.
        """
        if not isinstance(sample, dict):
            return "Other"
        if "conversations" in sample and isinstance(sample["conversations"], list):
            ok = True
            for msg in sample["conversations"]:
                if not (
                    (isinstance(msg, dict) and "role" in msg and "content" in msg) or
                    (isinstance(msg, dict) and "from" in msg and "value" in msg)
                ):
                    ok = False
                    break
            if ok:
                return "SFT Multi-Round"
        if "instruction" in sample and "output" in sample:
            if isinstance(sample["instruction"], str) and isinstance(sample["output"], str):
                if "input" not in sample or sample["input"] is None or isinstance(sample["input"], str):
                    return "SFT Single"
        pt_keys = {"text", "content", "sentence"}
        if len(sample) == 1:
            (k, v), = sample.items()
            if k in pt_keys and isinstance(v, str):
                return "PT"
        return "Other"

    # Storage选择
    if use_file_sys:
        from ..servicemanager import SampleFileStorage
        storage = SampleFileStorage(first_entry_file_name=json_file,cache_type="json")
        storage.step()  
    else:
        storage = None
    logger.debug(f"------------采样之前--------------------")
    samples,total = storage.rsample(mode="manual",k=sample_size)
    logger.debug(f"------------采样之后--------------------")

    type_list = [judge_type(s) for s in samples]
    counter = Counter(type_list)

    # total = len(samples)
    dist = {
        t: {"count": c, "ratio": round(c / total, 4)}
        for t, c in counter.items()
    }
    stats = {
        "total": total,
        "distribution": dist,
        "samples": samples
    }
    logger.debug(f"-------数据统计-------\n {stats}")
    return stats

def generate_pre_task_params(
    system_template: str,
    task_template: str,
    request: "ChatAgentRequest",
    param_funcs: Optional[Dict[str, Callable]] = None,
    pre_task_result: Any = None,
    memory = None,
    extral_param: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    logger.debug(f"[extral_param]:{extral_param}")
    if param_funcs:
        for param_name, func in param_funcs.items():
            try:
                import inspect
                sig        = inspect.signature(func)
                accepted   = sig.parameters.keys()

                kwargs = {}
                if "request" in accepted:
                    kwargs["request"] = request
                if "pre_task_result" in accepted:
                    kwargs["pre_task_result"] = pre_task_result
                if "memory" in accepted:
                    kwargs["memory"] = memory
                if extral_param and param_name in extral_param:
                    kwargs.update(extral_param[param_name])
                logger.debug(f"[param_name]:{param_name} \n [kwargs]:{kwargs}")  
                params[param_name] = func(**kwargs)
            except Exception as e:
                print(f"[generate_pre_task_params] {func.__name__} failed: {e}")
                params[param_name] = None
    return {
        "templates": [
            {
                "name": system_template,
                "params": {}
            },
            {
                "name": task_template,
                "params": params
            }
        ]
    }

# if __name__ == "__main__":
#     print(local_tool_for_sample(json_file="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json"))