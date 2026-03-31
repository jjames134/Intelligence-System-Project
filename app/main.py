from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from app.model import model_ml, model_dl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app = FastAPI()
app.include_router(model_ml.router)
app.include_router(model_dl.router)
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
    csv_path2 = os.path.join(BASE_DIR, "data", "dataset1.csv")
    df = pd.read_csv(csv_path2)

    # ตารางก่อน
    table_html_before = df.to_html(classes="table table-striped", index=False)

    # Summary dataset
    rows, cols = df.shape
    summary_info = {
        "rows": rows,
        "columns": cols,
        "total_values": rows * cols
    }

    # dtype
    dtype_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str)
    })
    table_html_dtype = dtype_summary.to_html(classes="table table-striped", index=False)

    # null summary
    null_summary = pd.DataFrame({
        "column": df.columns,
        "null_count": df.isnull().sum(),
        "has_null": df.isnull().any()
    })
    table_html_null = null_summary.to_html(classes="table table-striped", index=False)
    # copy เพื่อไม่กระทบ df2 เดิม
    df_clean = df.copy()

    # ลบคอลัมน์ที่ null เยอะ
    df_clean = df_clean.drop(columns=["poverty_rate", "gini_index"], errors="ignore")

    # ลบคอลัมน์ซ้ำซ้อน
    df_clean = df_clean.drop(columns=["country"], errors="ignore")

    # ลบแถวที่มี null ในคอลัมน์สำคัญ
    cols_to_check = [
      "gdp",
      "gdp_per_capita",
      "income_top1",
      "income_top10",
      "income_bottom50"
    ]

    df_clean = df_clean.dropna(subset=cols_to_check)

    # จัดการปี
    df_clean["year_class"] = df_clean["year"] - df_clean["year"].min()

    # ลบคอลัมน์ซ้ำซ้อน
    df_clean = df_clean.drop(columns=["year"], errors="ignore")

    # จัดการiso_code
    le = LabelEncoder()
    df_clean["iso_code"] = le.fit_transform(df_clean["iso_code"].astype(str))

    table_html_after = df_clean.to_html(
    classes="table table-striped",
    index=False,
    float_format="%.0f"   # ไม่มี e+
)
    return templates.TemplateResponse(
        request,
        "theory_ml.html",
        {
            "table_html_before": table_html_before,
            "table_html_null": table_html_null,
            "table_html_dtype": table_html_dtype,
            "summary_info": summary_info,
            "table_html_after": table_html_after
        }
    )

@app.get("/theory_dl")
async def pageDL(request: Request):

    # ================= LOAD =================
    csv_path = os.path.join(BASE_DIR, "data", "dataset2.csv")
    df = pd.read_csv(csv_path)

    # ================= BEFORE =================
    table_html_before = df.head(100).to_html(classes="table table-striped", index=False)

    rows, cols = df.shape
    summary_info = {
        "rows": rows,
        "columns": cols,
        "total_values": rows * cols
    }

    dtype_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str)
    })
    table_html_dtype = dtype_summary.to_html(classes="table table-striped", index=False)

    null_summary = pd.DataFrame({
        "column": df.columns,
        "null_count": df.isnull().sum(),
        "has_null": df.isnull().any()
    })
    table_html_null = null_summary.to_html(classes="table table-striped", index=False)

    # ================= CLEANING =================
    df_clean = df.copy()

    # 1. ลบ missing
    df_clean = df_clean.dropna()

    # 2. Encoding (categorical → numeric)
    label_encoders = {}
    categorical_cols = [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work"
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le


    # ================= AFTER =================
    table_html_after = df_clean.head(100).to_html(
        classes="table table-success table-striped",
        index=False
    )

    # ================= RETURN =================
    return templates.TemplateResponse(
        request,
        "theory_dl.html",
        {
            "table_html_before": table_html_before,
            "table_html_after": table_html_after,
            "table_html_null": table_html_null,
            "table_html_dtype": table_html_dtype,
            "summary_info": summary_info
        }
    )


