import os, asyncio
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from pathlib import Path
CONFIG_PATH = (Path(__file__).resolve().parent / ".." / "dataflow" / "agent" /
               "agentrole" / "resources" / "ChatAgentYaml.yaml").resolve()
from dataflow.agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.agent.taskcenter import TaskRegistry
import dataflow.agent.taskcenter.task_definitions
from dataflow.agent.servicemanager import AnalysisService, Memory
from dataflow.agent.toolkits import (
    ChatResponse,
    get_logger,
    setup_logging,
    main_print_logo,
    ChatAgentRequest,
)
main_print_logo()
logger = get_logger(__name__)
setup_logging("DEBUG")
memory = Memory()       

# class ChatAgentRequest(BaseModel):
#     language: str = Field(..., description="回答语言，例如 zh / en")
#     target: str   = Field(..., description="用户当前输入的内容")
#     model: str    = "deepseek-v3"
#     sessionKEY: str
#     temperature: Optional[float] = 0.7
#     max_tokens:   Optional[int]  = 1024


def _build_task_chain(language: str, with_execute: bool):
    tmpl = PromptsTemplateGenerator(language)
    router = TaskRegistry.get("conversation_router", tmpl)
    classify = TaskRegistry.get("data_content_classification", tmpl)
    rec      = TaskRegistry.get("recommendation_inference_pipeline", tmpl)
    if with_execute:
        exe  = TaskRegistry.get("execute_the_recommended_pipeline", tmpl)
        return [router, classify, rec, exe]
    return [router, classify, rec]

async def _run_service(req: ChatAgentRequest, with_execute: bool) -> ChatResponse:
    service = AnalysisService(
        config_path = CONFIG_PATH,
        tasks=_build_task_chain(req.language, with_execute),
        memory_entity=memory,
        request=req,
    )
    return await service.process_request()

app = FastAPI(title="Dataflow Agent Service")
@app.post("/recommend", response_model=ChatResponse)
async def recommend(req: ChatAgentRequest):
    return await _run_service(req, with_execute=False)

@app.post("/recommend_and_execute", response_model=ChatResponse)
async def recommend_and_execute(req: ChatAgentRequest):
    return await _run_service(req, with_execute=True)

if __name__ == "__main__":
    import uvicorn, json, sys
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        test_req = ChatAgentRequest(
            language="zh", #en 或者 zh
            target="帮我针对数据推荐一个预测的 pipeline,不需要去重的算子！",
            model="deepseek-v3",
            sessionKEY="dataflow_demo",
        )
        resp = asyncio.run(_run_service(test_req, with_execute=False))
        print(json.dumps(resp.dict(), ensure_ascii=False, indent=2))
    else:
        uvicorn.run("test_dataflow_agent:app", host="0.0.0.0", port=8000, reload=True)