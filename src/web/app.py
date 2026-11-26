import os
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from transformers import AutoModelForTokenClassification, AutoTokenizer

from address.region_repository import RegionRepository
from address.service import AddressAlignmentService
from configuration import config
from runner.predict import Predictor


class AddressAlignmentRequest(BaseModel):
    message: str = Field(min_length=1, description="地址文本")


class AddressAlignmentResponse(BaseModel):
    prov: str | None = None
    city: str | None = None
    district: str | None = None
    town: str | None = None
    detail: str | None = None
    name: str | None = None
    phone: str | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    # 服务启动时加载一次微调模型和分词器，后续请求复用同一个服务。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = config.MODELS_DIR / "best"
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到微调模型：{model_path}。请先训练并保存最佳权重。")

    tokenizer_path = config.PRE_TRAINED_DIR / "roberta-small-wwm-chinese-cluecorpussmall"
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    application.state.alignment_service = AddressAlignmentService(
        Predictor(device, model, tokenizer), RegionRepository()
    )
    yield


app = FastAPI(title="地址对齐", lifespan=lifespan)


@app.get("/", response_class=FileResponse)
def homepage():
    return FileResponse(config.ROOT_DIR / "templates" / "index.html", media_type="text/html")


@app.post("/address_alignment", response_model=AddressAlignmentResponse)
def handle_message(payload: AddressAlignmentRequest, request: Request):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="请输入地址文本")
    if "ADDRESS_DB_PASSWORD" not in os.environ:
        raise HTTPException(status_code=503, detail="请先在运行配置中设置 ADDRESS_DB_PASSWORD")

    # 同步路由由 FastAPI 在线程池中运行，模型推理和 MySQL 查询不会阻塞事件循环。
    service: AddressAlignmentService = request.app.state.alignment_service
    try:
        return service.align_address(message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="地区数据库暂时无法查询") from exc


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089)
