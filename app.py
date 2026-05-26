from fastapi import FastAPI
import uvicorn
import subprocess
import sys

from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import Response

from textSummarizer.pipeline.prediction import PredictionPipeline


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["web"])
async def index(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/train")
async def training():

    try:

        subprocess.run(
            [sys.executable, "main.py"],
            check=True
        )

        return Response("Training successful !!")

    except subprocess.CalledProcessError as e:

        return Response(
            f"Training failed with exit code {e.returncode}",
            status_code=500
        )


@app.post("/predict")
async def predict_route(text: str):

    try:

        obj = PredictionPipeline()

        summary = obj.predict(text)

        return summary

    except Exception as e:
        raise e


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8085
    )
