import logging
import os

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse


from config.log_config import uvicorn_logger # type: ignore
from utils.input_pattern import OpenAIinput # type: ignore
from utils.errors import TalosServerException # type: ignore
from code_generator.coder import CodeGenerator # type: ignore

logging.config.dictConfig(uvicorn_logger)

codegen = CodeGenerator()

app = FastAPI(
    title="TalosApi",
    description="Talos is an opensource implemetation for code completion server",
    docs_url="/",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

@app.exception_handler(TalosServerException)
async def fauxpilot_handler(request: Request, exc: TalosServerException):
    return JSONResponse(
        status_code=400,
        content=exc.json()
    )

 
@app.post("/v1/engines/codegen/completions", status_code=200)
@app.post("/v1/completions", status_code=200) 
async def completions(data: OpenAIinput):
    data = data.dict()
    try:
        content = codegen(data=data)
    except codegen.TokensExceedsMaximum as E:
        raise TalosServerException(
            message=str(E),
            type="invalid_request_error",
            param=None,
            code=None,
        )

    if data.get("stream") is not None:
        return EventSourceResponse(
            content=content,
            status_code=200,
            media_type="text/event-stream"
        )
    else:
        return Response(
            status_code=200,
            content=content,
            media_type="application/json"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5050)
    