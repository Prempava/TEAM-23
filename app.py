from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import joblib

# Internal modules
from .parameter import generate_floorplan
from .material import estimate_materials
from .image import generate_building_images
from .cost import estimate_cost
from explainer import generate_ai_explanation

app = FastAPI(title="AI-Based Construction Planning System")

# Base directory
BASE = Path(__file__).resolve().parents[1]

# Load ML model and label encoder
model = joblib.load(BASE / "models" / "classifier.joblib")
label_encoder = joblib.load(BASE / "models" / "label_encoder.joblib")

# Serve generated images folder
app.mount(
    "/generated_images",
    StaticFiles(directory="generated_images"),
    name="generated_images"
)


@app.get("/")
def home():
    return {
        "message": "AI Construction Planning API is running",
        "usage": "Open /docs to test the application"
    }


@app.post("/predict", response_class=HTMLResponse)
def predict_design(
    area_m2: float = Form(...),
    slope_percent: float = Form(...),
    bearing_capacity_kpa: float = Form(...),
    project_requirement: str = Form(...),
    plot_shape: str = Form(...),
    num_floors: int = Form(...),
    budget_usd: float = Form(...)
):
    """
    End-to-end AI construction planning:
    - ML building type prediction
    - Floorplan generation
    - Material estimation
    - Cost estimation
    - AI image generation
    """

    # Prepare input for ML model
    X = pd.DataFrame([{
        "area_m2": area_m2,
        "slope_percent": slope_percent,
        "bearing_capacity_kpa": bearing_capacity_kpa,
        "project_requirement": project_requirement,
        "plot_shape": plot_shape,
        "num_floors": num_floors,
        "budget_usd": budget_usd
    }])

    # ML prediction
    numeric_pred = model.predict(X)[0]
    confidence = float(model.predict_proba(X)[0].max())
    predicted_label = label_encoder.inverse_transform([numeric_pred])[0]

    # Generate floorplan
    floorplan = generate_floorplan(predicted_label, area_m2)

    # Estimate materials
    materials = estimate_materials(area_m2, num_floors)

    # Estimate cost
    total_cost = estimate_cost(materials)

    # Generate building images
    image_paths = generate_building_images(
        predicted_label,
        num_floors,
        area_m2,
        count=2
    )
    rooms_html = "".join(
        f"<li>{r['name'].capitalize()} – {r['area_m2']} m²</li>"
        for r in floorplan["rooms"]
    )

    images_html = "".join(
        f'<img src="/{p}" width="420" style="margin:10px; border:1px solid #ccc"/>'
        for p in image_paths
    )

    return f"""
    <html>
    <head>
        <title>AI Construction Result</title>
    </head>

    <body style="font-family: Arial, sans-serif; padding: 20px;">

    <h2>🏗️ AI Construction Planning Result</h2>

    <p><b>Recommended Building Type:</b> {predicted_label.upper()}</p>
    <p><b>Confidence Level:</b> {confidence:.2%}</p>

    <hr>

    <h3>📏 Land & Project Details</h3>
    <ul>
        <li>Total Area: {area_m2} m²</li>
        <li>Number of Floors: {num_floors}</li>
        <li>Plot Shape: {plot_shape}</li>
        <li>Project Requirement: {project_requirement}</li>
    </ul>

    <hr>

    <h3>📐 Floorplan</h3>
    <ul>
        {rooms_html}
    </ul>

    <hr>

    <h3>🧱 Material Estimation</h3>
    <ul>
        <li>Concrete Required: {materials['concrete_m3']} m³</li>
        <li>Steel Required: {materials['steel_kg']} kg</li>
        <li>Bricks Required: {materials['bricks_count']}</li>
    </ul>

    <h3>💰 Estimated Construction Cost</h3>
    <p><b>Total Cost:</b> ${total_cost}</p>

    <hr>

    <h3>🖼️ AI Generated Building Designs</h3>
    {images_html}

    <hr>

    <p style="color: gray; font-size: 12px;">
    ⚠️ This is an AI-generated early-stage construction plan.
    Final approval should be done by certified professionals.
    </p>

    </body>
    </html>
    """
@app.post("/predict/json")
def predict_design_json(
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
    total_cost = estimate_cost(materials)

    image_paths = generate_building_images(
        predicted_label,
        num_floors,
        area_m2,
        count=2
    )

    return {
        "predicted_building_type": predicted_label,
        "confidence": confidence,
        "input": {
            "area_m2": area_m2,
            "slope_percent": slope_percent,
            "bearing_capacity_kpa": bearing_capacity_kpa,
            "project_requirement": project_requirement,
            "plot_shape": plot_shape,
            "num_floors": num_floors,
            "budget_usd": budget_usd
        },
        "floorplan": floorplan,
        "materials": materials,
        "estimated_cost": total_cost,
        "generated_images": image_paths
    }
