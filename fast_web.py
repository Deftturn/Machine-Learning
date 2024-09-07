from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Web Devlopment
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

model = joblib.load(r"C:\Users\HP\desktop\my_folder\automationswithPython\UsingModels\iris_model.pkl")

app = FastAPI(debug=True)

# adding Css
def app_mount():
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
        name = "static"
    )

app_mount()

# templates
templates = Jinja2Templates(directory=r"C:\Users\HP\desktop\my_folder\automationswithPython\UsingModels\templates")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=IrisPrediction)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class] # type: ignore

    return templates.TemplateResponse("result.html", locals())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port=8000)
