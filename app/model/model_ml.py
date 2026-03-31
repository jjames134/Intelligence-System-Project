from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# ตั้งค่าไม่ให้ Matplotlib พยายามเปิดหน้าต่าง GUI บน Server
plt.switch_backend('Agg')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# -------------------- CONFIG & PATHS --------------------
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# ค้นหา Path ให้แม่นยำ (app/model/model_ml.py -> app/)
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_FILE_DIR)

templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

# -------------------- GLOBAL VARIABLES --------------------
ensemble = None
le_iso = None
X_columns = None
threshold_value = 0
metrics_data = {} # เก็บค่า Metrics ทั้งหมดไว้ในที่เดียวเพื่อให้ง่ายต่อการส่ง Context

# -------------------- INITIALIZE MODEL --------------------
def init_model():
    global ensemble, le_iso, X_columns, threshold_value, metrics_data

    if ensemble is not None:
        return

    # ชี้ไปที่ Root/app/data/dataset1.csv
    path = os.path.join(APP_DIR, "data", "dataset1.csv")
    if not os.path.exists(path):
        print(f"❌ ML Model Error: File not found at {path}")
        return

    df = pd.read_csv(path)

    # -------------------- CLEANING --------------------
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=["poverty_rate", "gini_index", "country"], errors="ignore")
    cols_to_check = ["gdp", "gdp_per_capita", "income_top1", "income_top10", "income_bottom50"]
    df_clean = df_clean.dropna(subset=cols_to_check)
    df_clean["year_class"] = df_clean["year"] - df_clean["year"].min()
    df_clean = df_clean.drop(columns=["year"], errors="ignore")

    le_iso = LabelEncoder()
    df_clean["iso_code"] = le_iso.fit_transform(df_clean["iso_code"].astype(str))

    # -------------------- HEATMAP --------------------
    corr = df_clean.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode()

    # -------------------- TARGET & SPLIT --------------------
    threshold_value = df_clean["income_top10"].median()
    df_clean["target"] = (df_clean["income_top10"] > threshold_value).astype(int)

    X = df_clean.drop(columns=["income_top10","target"])
    y = df_clean["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_columns = X_train.columns.tolist()

    # -------------------- ENSEMBLE MODEL --------------------
    m1 = LogisticRegression(max_iter=1000)
    m2 = DecisionTreeClassifier()
    m3 = RandomForestClassifier()

    ensemble = VotingClassifier(
        estimators=[("lr", m1), ("dt", m2), ("rf", m3)],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)

    # -------------------- METRICS & ROC --------------------
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    plt.figure()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    roc_base64 = base64.b64encode(buf.getvalue()).decode()

    # -------------------- CROSS VAL & COMPARISON --------------------
    cv = cross_val_score(ensemble, X, y, cv=5)
    
    model_comp = []
    for name, m in {"Logistic": m1, "Decision Tree": m2, "Random Forest": m3}.items():
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        model_comp.append((name, round(acc, 4)))

    # รวบรวมข้อมูลไว้ใน Dictionary เดียว
    metrics_data = {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc": round(auc_val, 4),
        "heatmap": heatmap_base64,
        "roc": roc_base64,
        "model_results": model_comp,
        "cv_mean": round(cv.mean(), 4),
        "cv_std": round(cv.std(), 4),
        "threshold": round(threshold_value, 2)
    }

init_model()

# -------------------- ROUTES --------------------
def get_ml_context(request: Request, extra_data=None):
    ctx = {"request": request}
    ctx.update(metrics_data)
    if extra_data:
        ctx.update(extra_data)
    return ctx

@router.get("/model")
async def model_page(request: Request):
    return templates.TemplateResponse(
        name="model_ml.html", 
        context=get_ml_context(request)
    )

@router.post("/predict")
async def predict_user(
    request: Request,
    population: float = Form(...),
    gdp: float = Form(...),
    gdp_per_capita: float = Form(...),
    income_top1: float = Form(...),
    income_bottom50: float = Form(...),
    year_class: int = Form(...)
):
    df_input = pd.DataFrame([{
        "population": population,
        "gdp": gdp,
        "gdp_per_capita": gdp_per_capita,
        "income_top1": income_top1,
        "income_bottom50": income_bottom50,
        "year_class": year_class,
        "iso_code": 0
    }])
    df_input = df_input[X_columns]
    pred = ensemble.predict(df_input)[0]
    proba = ensemble.predict_proba(df_input)[0][1]

    res = {
        "user_pred": "High Inequality" if pred == 1 else "Low Inequality",
        "probability": round(proba * 100, 2)
    }
    
    return templates.TemplateResponse(
        name="model_ml.html", 
        context=get_ml_context(request, res)
    )