import os

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
os.environ["HANLP_HOME"] = "./checkpoints/hanlp_cache"

import argparse
import time

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

import hanlp
from wdd.file_utils import logging
from wdd.model.ProcessTokModel import ProcessTokRequest, ProcessTokResponse
from wdd.TextProcessor import TextProcessor

# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有。

app = FastAPI(title="Hanlp Service", version="1.0", docs_url=None)

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"],
)  # 允许跨域的headers，可以用来鉴别来源等作用。
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# 使用本地的 Swagger UI 静态资源
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Custom Swagger UI",
        swagger_js_url="/static/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/5.9.0/swagger-ui.css",
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.get("/test")
async def test():
    """
    测试接口，用于验证服务是否正常运行。
    """
    return PlainTextResponse("success")


@app.post("/process_tok/", response_model=ProcessTokResponse)
async def process_tok(request: ProcessTokRequest):
    """
    处理中文分词。
    """
    response = ProcessTokResponse()
    # 记录开始时间
    start_time = time.time()

    try:
        tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        if len(request.dict_force) > 0:
            tokenizer.dict_force = request.dict_force
        tokens = tokenizer(request.text)
        response.tokens = tokens
    except Exception as ex:
        TextProcessor.log_error(ex)
        response.errcode = -1
        response.errmsg = f"处理失败：{str(ex)}"

    # 计算耗时
    elapsed = time.time() - start_time
    logging.info(f"分词生成完成，用时: {elapsed}")

    return response


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=8120, help="Port to run the server on"
    )
    args, unknown = parser.parse_known_args()

    try:

        uvicorn.run(
            app="api:app",
            host="0.0.0.0",
            port=args.port,
            workers=1,
            reload=False,
            log_level="info",
        )
    except Exception as e:
        TextProcessor.log_error(e)
        print(e)
        exit(0)
