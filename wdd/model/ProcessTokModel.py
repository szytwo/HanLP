from typing import Any

from pydantic import BaseModel, Field

from wdd.model.APIBaseModel import ResponseBaseModel


class ProcessTokRequest(BaseModel):
    text: str | list[str] = Field(
        ...,
        description="需要分词的文本，必填",
    )
    dict_force: list[str] = Field(
        default=[],
        description="强制自定义词条列表，例如：['我趣玩', '我趣玩AI', '数字人']",
    )

    class Config:
        json_schema_extra = {
            "description": "分词处理的请求体",
            "example": {
                "text": "欢迎使用我趣玩AI的数字人服务",
                "dict_force": ["我趣玩", "我趣玩AI", "数字人"],
            },
        }


class ProcessTokResponse(ResponseBaseModel):
    tok: list[str] = Field(
        default=[],
        description="分词结果",
    )
    pos: list[str] = Field(
        default=[],
        description="词性标注结果",
    )
    ner: Any = Field(
        default=[],
        description="命名实体识别结果",
    )
    dep: Any = Field(
        default=[],
        description="依存句法分析结果",
    )

    class Config:
        json_schema_extra = {
            "description": "分词处理的响应结果",
            "example": {
                "errcode": 0,
                "errmsg": "ok",
                "tok": ["我趣玩AI", "是", "数字人"],
            },
        }
