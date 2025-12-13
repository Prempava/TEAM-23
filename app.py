from fastapi import FastAPI, Form
import joblib
import pandas as pd
from pathlib import Path

from .parameter import generate_floorplan
from .material import estimate_materials

app = FastAPI(title="AI Construction Generator")

BASE = Path(__file__).resolve().parents[1]

model = joblib.load(BASE / "models" / "classifier.joblib")
label_encoder = joblib.load(BASE / "models" / "label_encoder.joblib")

@app.post("/predict")
def predict_design(
    area_m2: float = Form(...),
    slope_percent: float = Form(...),
    bearing_capacity_kpa: float = Form(...),
    project_requirement: str = Form(...),
    plot_shape: str = Form(...),
    num_floors: int = Form(...),
    budget_usd: float = Form(...)
):

    X = pd.DataFrame([{
        "area_m2": area_m2,
        "slope_percent": slope_percent,
        "bearing_capacity_kpa": bearing_capacity_kpa,
        "project_requirement": project_requirement,
        "plot_shape": plot_shape,
        "num_floors": num_floors,
        "budget_usd": budget_usd
    }])

    numeric_pred = model.predict(X)[0]
    confidence = float(model.predict_proba(X)[0].max())
    predicted_label = label_encoder.inverse_transform([numeric_pred])[0]

    floorplan = generate_floorplan(predicted_label, area_m2)
    materials = estimate_materials(area_m2, num_floors)

    return {
        "Predicted Building Type": predicted_label,
        "Confidence": round(confidence, 3),
        "Floorplan": floorplan,
        "Materials Required": materials
    }
