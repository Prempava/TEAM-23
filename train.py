import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parents[0]
DATA = BASE / "D:\hackathon\data\synthetic_data.csv"   
OUT = BASE / "models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
label_encoder = LabelEncoder()
df["label_numeric"] = label_encoder.fit_transform(df["label_template"])

joblib.dump(label_encoder, OUT / "label_encoder.joblib")

X = df[[
    "area_m2",
    "slope_percent",
    "bearing_capacity_kpa",
    "project_requirement",
    "plot_shape",
    "num_floors",
    "budget_usd"
]]

y = df["label_numeric"]  

cat_cols = ["project_requirement", "plot_shape"]
num_cols = ["area_m2", "slope_percent", "bearing_capacity_kpa", "num_floors", "budget_usd"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(eval_metric="mlogloss", n_estimators=200))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("Label classes:", label_encoder.classes_)
print(classification_report(y_test, pred))

joblib.dump(pipeline, OUT / "classifier.joblib")
print("Model saved to:", OUT / "classifier.joblib")
