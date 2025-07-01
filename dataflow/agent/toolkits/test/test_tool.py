from __future__ import annotations
import os
from typing import List, Dict, Sequence, Any, Union, Optional, Iterable, Mapping, Set, Callable
from pathlib import Path
import subprocess
from collections import defaultdict, deque

def _topological_sort(nodes: List[Dict[str, Any]],
                      edges: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Kahn algorithm – 返回 nodes 的拓扑顺序。
    若 DAG 不合法(有环)，抛出 ValueError。
    """
    id2node = {n["id"]: n for n in nodes}
    indeg = defaultdict(int)
    graph = defaultdict(list)

    for e in edges:
        src, dst = e["source"], e["target"]
        graph[src].append(dst)
        indeg[dst] += 1
        # 保证所有节点出现在 indeg
        indeg[src] += 0

    # 未出现在 edges 的孤立节点(无入度无出度)也需执行
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
        raise ValueError("Graph has cycle or missing edges; "
                         f"sorted {len(order)} / {len(id2node)} nodes.")

    return [id2node[nid] for nid in order]

def _write_shell_script(ordered_nodes: Iterable[Dict[str, Any]],
                        sh_path: str,
                        env_vars: Dict[str, str] | None = None) -> None:
    """
    生成 shell 脚本文件。
    """
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -e\n\n")
        # 环境变量
        if env_vars:
            f.write("# -------- env -------- #\n")
            for k, v in env_vars.items():
                f.write(f'export {k}="{v}"\n')
            f.write("\n")
        # 每一步
        f.write("# -------- pipeline -------- #\n")
        for step, node in enumerate(ordered_nodes):
            title = node["name"] if node.get("name") else node["id"]
            f.write(f'echo -e "\\033[32m===== [Step {step}] {title} =====\\033[0m"\n')
            f.write(f'{node["command"]}\n\n')
    os.chmod(sh_path, 0o755)

def local_tool_for_execute_the_recommended_pipeline(
    pipeline_config: Dict[str, Any],
    *,
    sh_path: str = "run_pipeline.sh",
    env_vars: Dict[str, str] | None = None,
    execute: bool = True,
    dry_run: bool = False,
) -> str:
    """
    主入口。

    参数
    ----
    pipeline_config : dict          # 满足题干所给格式
    sh_path         : str           # 生成的 shell 文件保存路径
    env_vars        : dict|None     # 需要写入 export 的环境变量
    execute         : bool          # 生成后是否立即 bash 执行
    dry_run         : bool          # True 时不落盘、不执行，只返回脚本内容

    返回值
    ----
    sh_content : str  生成的完整脚本文本
    """
    ctx = pipeline_config["info"]["context"]
    nodes, edges = ctx["nodes"], ctx.get("edges", [])
    ordered_nodes = _topological_sort(nodes, edges)

    # 拼接脚本文本
    lines: List[str] = ["#!/usr/bin/env bash", "set -e", ""]
    if env_vars:
        lines.append("# -------- env -------- #")
        for k, v in env_vars.items():
            lines.append(f'export {k}="{v}"')
        lines.append("")
    lines.append("# -------- pipeline -------- #")
    for step, node in enumerate(ordered_nodes):
        title = node.get("name", node["id"])
        lines.append(f'echo -e "\\033[32m===== [Step {step}] {title} =====\\033[0m"')
        lines.append(node["command"])
        lines.append("")
    sh_content = "\n".join(lines)

    if dry_run:
        return sh_content

    # 写文件
    _write_shell_script(ordered_nodes, sh_path, env_vars)

    # 执行脚本
    if execute:
        print(f"[INFO] Running pipeline: bash {sh_path}")
        # subprocess.run(["bash", sh_path], check=True)
    return sh_content

if __name__ == "__main__":
    # 1. 你的原始数据（只取 info 字段）
    raw_data = {
        "id": "584ca7a7-a242-40d9-8a30-9be91207cb45",
        "name": "Analysis Agent",
        "info": {
            "context": {
                "edges": [
                    {"source": "node1", "target": "node2"},
                    {"source": "node2", "target": "node3"},
                    {"source": "node3", "target": "node4"},
                    {"source": "node4", "target": "node5"},
                    {"source": "node5", "target": "node6"},
                    {"source": "node6", "target": "node7"},
                    {"source": "node7", "target": "node8"},
                    {"source": "node8", "target": "node9"}
                ],
                "reason": "This pipeline is designed to process and enhance math and science problems through a series of sequential steps. It starts with generating new questions, filtering for correctness, classifying difficulty and category, separating data with and without answers, generating pseudo-answers for incomplete data, and finally filtering answers based on format, length, and n-gram uniqueness to ensure high-quality output.",
                "outputs": "",
                "nodes": [
                    {
                        "name": "QuestionGenerator",
                        "type": "generator",
                        "description": "基于现有的问题数据，每个问题合成1-5个新问题",
                        "command": "python ReasoningPipeline/code/QuestionGenerator.py --yaml_path ReasoningPipeline/yaml/QuestionGenerator.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node1"
                    },
                    {
                        "name": "MathProblemFilter",
                        "type": "filter",
                        "description": "检查每个问题的正确性",
                        "command": "python process.py --config configs/process/math/pipeline_Q/test_process_math_step2.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node2"
                    },
                    {
                        "name": "QuestionDifficultyClassifier",
                        "type": "generator",
                        "description": "为每个问题确定一个难度分数标签",
                        "command": "python ReasoningPipeline/code/QuestionDifficultyClassifier.py --yaml_path ReasoningPipeline/yaml/QuestionDifficultyClassifier.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node3"
                    },
                    {
                        "name": "QuestionCategoryClassifier",
                        "type": "generator",
                        "description": "将所有问题分类到7个大类别，以及每个大类别下的若干的小类别",
                        "command": "python ReasoningPipeline/code/QuestionCategoryClassifier.py --yaml_path ReasoningPipeline/yaml/QuestionCategoryClassifier.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node4"
                    },
                    {
                        "name": "AnswerPipelineRoot",
                        "type": "generator",
                        "description": "用于检查数据是否包含Answer、groundtruth，并分离有答案和没答案的数据，方便后续分别处理",
                        "command": "python ReasoningPipeline/code/AnswerPipelineRoot.py --yaml_path ReasoningPipeline/yaml/AnswerPipelineRoot.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node5"
                    },
                    {
                        "name": "PseudoAnswerGenerator",
                        "type": "generator",
                        "description": "为没有答案的数据依据模型多次回答 voting 生成伪答案",
                        "command": "python ReasoningPipeline/code/PseudoAnswerGenerator.py --yaml_path ReasoningPipeline/yaml/PseudoAnswerGenerator.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node6"
                    },
                    {
                        "name": "AnswerFormatterFilter",
                        "type": "processor",
                        "description": "按照给定的格式，基于规则过滤掉不符合格式要求的数据",
                        "command": "python process.py --config configs/process/math/pipeline_GT/text_process_reasoner_formatfilter.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node7"
                    },
                    {
                        "name": "AnswerTokenLengthFilter",
                        "type": "processor",
                        "description": "过滤掉Answer长度不合适的数据",
                        "command": "python process.py --config configs/process/math/pipeline_withoutGT/text_process_reasoner_lengthfilter.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node8"
                    },
                    {
                        "name": "AnswerNgramFilter",
                        "type": "processor",
                        "description": "对Q和A一起构成的字符串进行 n-gram 去重",
                        "command": "python process.py --config configs/process/math/pipeline_withoutGT/text_process_reasoner_ngramfilter.yaml",
                        "required": True,
                        "depends_on": [],
                        "id": "node9"
                    }
                ],
                "name": "2429661e-5844-47dd-890f-db8cbbc3e58e_pipeline"
            },
            "response": "根据您提供的pipeline结构和目标（处理数学和科学问题，通过生成、过滤、分类、答案生成和优化等步骤），这是一个设计良好的**数学/科学问题处理流水线**。..."
        }
    }

    # 2. 只取 info 字段作为 pipeline_config
    pipeline_config = {"info": raw_data["info"]}

    # 3. 可选：定义环境变量
    env_vars = {"APIKEY": "sk-", "HF": "https://hf-mirror.com"}

    # 4. 调用主函数（dry_run=True 只生成脚本内容，不实际写文件/执行）
    sh_content = local_tool_for_execute_the_recommended_pipeline(
        pipeline_config,
        sh_path="test_pipeline.sh",
        env_vars=env_vars,
        execute=False,     # 不实际执行shell
        dry_run=True       # 只输出脚本内容
    )

    print("==== 生成的 Shell 脚本内容 ====")
    print(sh_content)