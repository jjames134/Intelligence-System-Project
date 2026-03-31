from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app = FastAPI()


@app.get("/root")
def root():
    return {"m":"Hello World"}

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
    request,
    "index.html",
    {
        "result": None
    }
)


@app.get("/theory_ml")
async def pageML(request: Request):
    # อ่านไฟล์ 
    csv_path = os.path.join(BASE_DIR, "data", "dataset1(origin).csv")
    df = pd.read_csv(csv_path)
    # แปลง DataFrame เป็น HTML
    table_html_before = df.to_html(classes="table table-striped", index=False)
    
    return templates.TemplateResponse(
    request,"theory_ml.html", 
    {"table_html_before": table_html_before})

@app.get("/theory_dl")
async def pageDL(request: Request):
    # อ่านไฟล์ 
    csv_path = os.path.join(BASE_DIR, "data", "netflix_titles.csv")
    df = pd.read_csv(csv_path)
    # แปลง DataFrame เป็น HTML
    table_html_before = df.to_html(classes="table table-striped", index=False)
    
    return templates.TemplateResponse(
    request,"theory_dl.html", 
    {"table_html_before": table_html_before})