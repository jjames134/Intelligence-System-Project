from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# -------------------- CONFIG --------------------
router = APIRouter(prefix="/ml", tags=["Machine Learning"])
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# -------------------- GLOBAL VARIABLES --------------------
df_clean = None
ensemble = None
le_iso = None
X_columns = None
threshold_value = 0
tp = tn = fp = fn = 0
accuracy = precision = recall = f1 = auc = 0
heatmap_base64 = roc_base64 = ""
model_results = []
cv_mean = cv_std = 0

# -------------------- INITIALIZE MODEL --------------------
def init_model():
    global df_clean, ensemble, le_iso, X_columns
    global tp, tn, fp, fn
    global accuracy, precision, recall, f1, auc
    global heatmap_base64, roc_base64
    global model_results, cv_mean, cv_std
    global threshold_value

    # Load dataset
    path = os.path.join(BASE_DIR, "data", "dataset1.csv")
    df = pd.read_csv(path)

    # -------------------- CLEANING --------------------
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=["poverty_rate", "gini_index", "country"], errors="ignore")

    cols_to_check = ["gdp", "gdp_per_capita", "income_top1", "income_top10", "income_bottom50"]
    df_clean = df_clean.dropna(subset=cols_to_check)

    df_clean["year_class"] = df_clean["year"] - df_clean["year"].min()
    df_clean = df_clean.drop(columns=["year"], errors="ignore")

    # Label Encoding
    le_iso = LabelEncoder()
    df_clean["iso_code"] = le_iso.fit_transform(df_clean["iso_code"].astype(str))

    # -------------------- HEATMAP --------------------
    corr = df_clean.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    heatmap_base64 = base64.b64encode(buf.read()).decode()

    # -------------------- TARGET --------------------
    threshold_value = df_clean["income_top10"].median()
    df_clean["target"] = (df_clean["income_top10"] > threshold_value).astype(int)

    X = df_clean.drop(columns=["income_top10","target"])
    y = df_clean["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_columns = X_train.columns.tolist()

    # -------------------- MODELS --------------------
    m1 = LogisticRegression(max_iter=1000)
    m2 = DecisionTreeClassifier()
    m3 = RandomForestClassifier()

    ensemble = VotingClassifier(
        estimators=[("lr", m1), ("dt", m2), ("rf", m3)],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)

    # -------------------- METRICS --------------------
    y_pred = ensemble.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # -------------------- ROC --------------------
    y_prob = ensemble.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    roc_base64 = base64.b64encode(buf.read()).decode()

    # -------------------- CROSS VALIDATION --------------------
    cv = cross_val_score(ensemble, X, y, cv=5)
    cv_mean = cv.mean()
    cv_std = cv.std()

    # -------------------- MODEL COMPARISON --------------------
    model_results.clear()
    for name, model in {"Logistic": m1, "Decision Tree": m2, "Random Forest": m3}.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        model_results.append((name, round(acc,4)))


# Initialize when import
init_model()

# -------------------- ROUTES --------------------
@router.get("/model")
def model_page(request: Request):
    return templates.TemplateResponse(
        request,
        "model_ml.html",
        {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round(accuracy,4),
            "precision": round(precision,4),
            "recall": round(recall,4),
            "f1": round(f1,4),
            "auc": round(auc,4),
            "heatmap": heatmap_base64,
            "roc": roc_base64,
            "model_results": model_results,
            "cv_mean": round(cv_mean,4),
            "cv_std": round(cv_std,4),
            "threshold": round(threshold_value,2)
        }
    )

@router.post("/predict")
def predict_user(
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

    return templates.TemplateResponse(
        request,
        "model_ml.html",
        {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round(accuracy,4),
            "precision": round(precision,4),
            "recall": round(recall,4),
            "f1": round(f1,4),
            "auc": round(auc,4),
            "heatmap": heatmap_base64,
            "roc": roc_base64,
            "model_results": model_results,
            "cv_mean": round(cv_mean,4),
            "cv_std": round(cv_std,4),
            "threshold": round(threshold_value,2),
            "user_pred": "High Inequality" if pred==1 else "Low Inequality",
            "probability": round(proba*100,2)
        }
    )