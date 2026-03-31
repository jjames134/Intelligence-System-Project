from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
import numpy as np
import io, base64
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

router = APIRouter(prefix="/dl", tags=["Deep Learning"])

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ================= GLOBALS =================
df_clean = None
model = None
scaler = None
label_encoders = {}
X_columns = None
metrics_data = {}

roc_base64 = ""
heatmap_base64 = ""

# ================= INIT MODEL =================
def init_model():
    global df_clean, model, scaler, label_encoders, X_columns
    global metrics_data, roc_base64, heatmap_base64

    path = os.path.join(BASE_DIR, "data", "dataset2.csv")
    df = pd.read_csv(path).dropna()

    df_clean = df.copy()

    categorical_cols = ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le

    df_clean["salary_class"] = (df_clean["salary"] > df_clean["salary"].median()).astype(int)

    X = df_clean.drop(columns=["salary", "salary_class"])
    y = df_clean["salary_class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_columns = X_train.shape[1]

    model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ===== METRICS =====
    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_pred_prob)

    # ===== ROC PLOT =====
    fpr, tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    global roc_base64
    roc_base64 = base64.b64encode(buf.read()).decode()

    # ===== HEATMAP =====
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    global heatmap_base64
    heatmap_base64 = base64.b64encode(buf.read()).decode()

    metrics_data.update({
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": round(accuracy,4), "precision": round(precision,4),
        "recall": round(recall,4), "f1": round(f1,4), "auc": round(auc,4)
    })

init_model()

# ================= ROUTES =================
@router.get("/model")
def model_page(request: Request):
    return templates.TemplateResponse(
        request, "model_dl.html",
        {
            **metrics_data,
            "roc": roc_base64,
            "heatmap": heatmap_base64
        }
    )

@router.post("/predict")
def predict_user(
    request: Request,
    job_title: str = Form(...),
    experience_years: float = Form(...),
    education_level: str = Form(...),
    skills_count: int = Form(...),
    industry: str = Form(...),
    company_size: str = Form(...),
    location: str = Form(...),
    remote_work: str = Form(...),
    certifications: int = Form(...)
):
    try:
        raw_data = [job_title, experience_years, education_level,
                    skills_count, industry, company_size,
                    location, remote_work, certifications]

        cat_indices = [0,2,4,5,6,7]
        cat_names = ["job_title","education_level","industry","company_size","location","remote_work"]
        for idx, col_name in zip(cat_indices, cat_names):
            raw_data[idx] = label_encoders[col_name].transform([raw_data[idx]])[0]

        data_scaled = scaler.transform([raw_data])
        pred_prob = model.predict_proba(data_scaled)[0][1]
        pred_label = "💰 High Salary" if pred_prob > 0.5 else "📉 Low Salary"
        user_pred = f"{pred_label} ({round(pred_prob*100,2)}%)"

    except Exception as e:
        print(f"Prediction Error: {e}")
        user_pred = "❌ Prediction Error"

    return templates.TemplateResponse(
        request, "model_dl.html",
        {
            **metrics_data,
            "roc": roc_base64,
            "heatmap": heatmap_base64,
            "user_pred": user_pred
        }
    )