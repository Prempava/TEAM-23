🏗️ AI-Based Construction Planning System (AI-CPS)

✨ Overview

The AI-Based Construction Planning System (AI-CPS) is an intelligent platform designed to

automate and simplify the critical initial phases of construction planning. By leveraging

Machine Learning (ML) and generative rule-based logic, this system transforms raw site data

(land area, soil properties) into actionable design recommendations and preliminary cost estimates.



This project reduces the dependency on multiple experts for initial feasibility studies,

drastically cuts down planning time, and empowers users to make data-driven decisions on their construction projects.



🚀 Key Features



->Intelligent Building Type Prediction: Uses a trained ML model to classify the most

structurally and geographically suitable building type (e.g., Low-Rise Residential, High-Rise Commercial, Industrial Warehouse).



->Automated Conceptual Floorplan: Generates a logical 2D floorplan layout based on 

the predicted building type and user-defined room requirements using parametric design principles.



->Preliminary Material Estimation: Calculates the required bulk quantities of primary 

construction materials (concrete, steel, masonry) based on the generated design and structural inferences from soil data.



->Easy-to-Use API: Provides a clean, modern API endpoint via FastAPI for seamless integration into web applications.



💡 Motivation & Problem Solved



Traditional construction planning is often manual, expensive, and inaccessible to individuals without specialized technical knowledge. Common challenges include:



->Complexity of Site Assessment: Non-experts struggle to interpret geotechnical data 

(like Soil Bearing Capacity) to determine if a site can support a multi-story structure.



->High Upfront Costs: Initial consultation fees with architects and engineers for feasibility can be prohibitive.



->Time-Consuming Iteration: Manual generation and revision of initial concepts can take weeks.



*AI-CPS automates this decision-making process, providing rapid, data-backed initial feasibility results in minutes.



🛠️ System Architecture

The system operates as a modular, three-stage pipeline linked via a FastAPI server.



[Image Tag: System Architecture Diagram showing data flow:

 User Input -> ML Classification Module (XGBoost) -> Floorplan Generator -> Material Estimation Module -> Final Output/API]



->ML Classification Module: Takes numerical inputs (Land Area, Slope, Soil Bearing Capacity, etc.)

and uses the pre-trained XGBoost model to predict the optimal Building Type.



->Floorplan Generator: Uses the predicted Building Type and user-defined rooms to generate a conceptual

layout based on a rule-based/parametric engine (e.g., maximizing sunlight exposure, ensuring necessary room adjacencies).



->Material Estimation Module: Applies engineering unit rates and formulas based on the generated floorplan 

area and foundation requirements (inferred from SBC) to output a preliminary Bill of Quantities.



💻 Technologies Used

Component,Technology,Rationale

Programming Language,Python 3.x,Robust ecosystem for ML and data processing.

Machine Learning,XGBoost,High performance and accuracy for structured classification data.

Backend Framework,FastAPI,"Modern, fast (ASGI) framework for production-ready API serving."

Data Processing,"Pandas, NumPy",Efficient data handling and numerical computation.

Documentation,ReportLab (optional),Library for generating structured PDF reports from system outputs.

1. 📊 Data Generation Module (datageneration.py)

This module is responsible for creating the synthetic training data required for the Machine Learning (ML)
Classification component of the AI-Based Construction Planning System (AI-CPS).
The generated data simulates various combinations of land parameters, soil conditions,
 and project requirements, linking them to an optimal pre-defined Building Template (Label).

📁 File Location and Output

Script: datageneration.py
Output File: The script automatically creates and saves the generated dataset to: ../data/synthetic_data.csv

⚙️ How the Data is Generated
The script generates 2000 rows of simulated construction project data, featuring the following steps:

1.Land Measurements: Random plot dimensions (l, w) and area are generated.

2.Soil Properties: A random soil_type is chosen (clay, sand, loam, laterite). A base bearing_capacity_kpa is
mapped to the soil type and then randomized to simulate real-world variance.

3.Project Parameters: Random values for slope_percent, project_requirement (house, shop, warehouse), and num_floors are assigned.

4.Labeling Logic (choose_template function): This critical function implements simple rule-based expert logic to assign the final 
target variable (label_template).

    ->Large areas ( > 1200 m^2) or explicit warehouse requests are labeled as "warehouse".
    
    ->Poor soil bearing capacity ( < 100 kpa) combined with small area ( < 200 m^2) defaults to "single_storey_house".
    
    ->Moderate soil ( \ge 100 kpa) and medium area ( > 200 m^2) for a house request defaults to "duplex".


📈 Data Schema (Columns)

The generated CSV file contains the following columns, which will serve as the features (inputs) and the label (output) for the ML model:

Column Name,Type,Description,Role in ML
plot_length_m,Numeric,Length of the plot in meters.,Feature
plot_width_m,Numeric,Width of the plot in meters.,Feature
area_m2,Numeric,Total area of the plot.,Feature
slope_percent,Numeric,Gradient of the land.,Feature
soil_type,Categorical,"Type of soil (e.g., clay, sand).",Feature (needs encoding)

bearing_capacity_kpa,Numeric,Soil's ability to support weight (kN/m2).,Critical Feature
project_requirement,Categorical,"User's initial goal (house, shop, warehouse).",Feature (needs encoding)
num_floors,Integer,User-requested number of floors.,Feature
label_template,Categorical,Optimal Building Type assigned by rules.,Target Variable (Label)
Other Columns,Numeric/Cat,"Plot shape, orientation, budget (auxiliary data).",Auxiliary

▶️ Execution

   Prerequisites
   
   ->Ensure you have a modern Python environment installed.

Steps to Generate Data:

1.Navigate to the directory containing datageneration.py.

2.Run the script using the Python interpreter:
      python datageneration.py
      
Expected Output

The script will successfully create the output directory (data/) if it doesn't exist and print a confirmation message:

Synthetic data written to: [Path/to/your/project]/data/synthetic_data.csv

Next Step: Once the synthetic_data.csv file is generated, the next stage is Data Preprocessing and Model Training using the XGBoost classifier.

2. 🧠 ML Training Module (train_classifier.py)
   
This module executes the crucial second stage of the AI-CPS: Preprocessing the synthetic data and training the XGBoost Classification Model.

The goal is to teach the model how to map the raw land and project parameters to the optimal Building Template, as defined by the logic in the data generation step.

⚙️ Module Functionality

The script performs the following sequence of operations:

1.Data Ingestion: Reads the synthetic_data.csv file generated in the previous step.

2.Label Encoding: Converts the categorical target variable (label_template - e.g., 'duplex', 'warehouse') into numerical labels (label_numeric) for the classifier.

     ->The fitted LabelEncoder is immediately saved (label_encoder.joblib) for later use in production, ensuring predictions can be converted back to meaningful names.
     
3.Feature Selection: Defines the necessary input features (X) and the target variable (y).

4.Data Splitting: Splits the dataset into training (80%) and testing (20%) sets, using stratify=y to ensure an even distribution of building types in both sets.

5.Preprocessing Pipeline: Constructs a ColumnTransformer and integrates it into a Pipeline.

      -Categorical Features (cat_cols): Use OneHotEncoder to convert categorical text data (project_requirement, plot_shape) into a numerical format readable by the model.
      
      -Numerical Features (num_cols): Passed through without transformation.
      
6.Model Training: The XGBClassifier is trained using the preprocessed training data.

7.Evaluation: Predicts on the test set and prints a detailed classification_report.

8.Model Serialization: Saves the entire preprocessing and model pipeline (classifier.joblib) for deployment in the FastAPI backend.

💻 Dependencies

This script relies heavily on standard data science libraries:

-pandas: Data manipulation.

-joblib: Serialization (saving/loading models and encoders).

-sklearn: Data splitting, preprocessing tools (LabelEncoder, OneHotEncoder, ColumnTransformer, Pipeline).

-xgboost: The machine learning classifier.

🛠️ Key Components Saved

The training module saves two critical files to the models/ directory:
File Name,Content,Purpose
label_encoder.joblib,The fitted sklearn.preprocessing.LabelEncoder.,"Mandatory for converting the model's numerical output (0, 1, 2...) back into the original building names (e.g., 'duplex')."
classifier.joblib,The complete sklearn.pipeline.Pipeline object.,The trained XGBoost model combined with all necessary preprocessing steps (One-Hot Encoding). This single file is ready for deployment.

▶️ Execution and Results

1. Execute the Training Script

Ensure you are in the directory containing the script and the synthetic_data.csv file is correctly located (as defined by the DATA path).
python train_classifier.py

2. Expected Output

The script will first print the list of trained building classes, followed by the comprehensive performance report:
Label classes: ['duplex' 'shop_small' 'single_storey_house' 'warehouse']
              precision    recall  f1-score   support

           0       0.96      0.97      0.96       500  # Example for 'duplex'
           1       0.98      0.95      0.96       500  # Example for 'shop_small'
           2       0.94      0.93      0.94       500  # Example for 'single_storey_house'
           3       0.99      0.99      0.99       500  # Example for 'warehouse'

    accuracy                           0.97      2000
   macro avg       0.97      0.97      0.97      2000
weighted avg       0.97      0.97      0.97      2000

Model saved to: [Path/to/project/models/classifier.joblib]

Note: The actual output will vary based on the random seed and data generation, but high scores are expected given the rule-based data generation.

3. 📐 Floorplan Generator Module
 (parametric_generator.py)

This module implements the Rule-Based/Parametric Design Logic used in the AI-CPS. Its primary function is to translate the Predicted Building Type (output from the ML model) and the Allocated Area into a conceptual, spatial breakdown of the floorplan.

This output forms the basis for the Material Estimation module and provides the user with the first visual idea of their project layout.

⚙️ Module Functionality

The generate_floorplan function executes the following steps:

1.Template Lookup: Retrieves the list of required rooms for the template_name (e.g., "duplex" requires living, kitchen, two bedroom, and bath).

2.Equal Area Allocation: For simplicity, the total available area (area_m2) is divided equally among all required rooms.

3.Dimension Estimation: Calculates the approximate side length of the entire structure ($\sqrt{\text{Area}}$).

4.Room Dimensioning: For each room, approximate dimensions (width and height) are calculated using a fixed aspect ratio (derived from $\text{Area} = \text{Width} \times \text{Height}$, where $\text{Height} \approx 1.5 \times \text{Width}$ is implied by $W = \sqrt{A/1.5}$).

Output: A structured JSON dictionary containing the total area, approximate overall side length, and a list of all rooms with their allocated areas and estimated dimensions.

🛠️ Key Data Structure (TEMPLATES)

The core of the rule-based system is the TEMPLATES dictionary, which defines the fundamental requirements for each building type:

Template Name,Required Rooms,Minimum Total Area (m2)
single_storey_house,"living, kitchen, bedroom, bath",50
duplex,"living, kitchen, 2x bedroom, bath",120
warehouse,"open_space, office, toilet",300
shop_small,"retail, storage, toilet",40

📐 Mathematical Logic

The script uses basic area formulas for dimension estimation:

1.Total Side Estimate:$$\text{Side} = \sqrt{\text{Area}_{\text{Total}}}$$

2.Room Dimension Estimate (Assuming Aspect Ratio $H/W \approx 1.5$):For a given room area $A_{room}$, the dimensions are approximated:
       $$\text{Width} \approx \sqrt{A_{\text{room}} / 1.5}$$
                                                                                   $$\text{Height} \approx A_{\text{room}} / \text{Width}$$


Note: The current implementation focuses on area distribution; the final floorplan geometry (placement, adjacency) would be handled by a more complex layout algorithm in a production environment.

▶️ Execution Example

To test the module independently, run the script directly:

     python src/parametric_generator.py

Example JSON Output (for single_storey_house with $120 m^2$)
       
{
  "template": "single_storey_house",
  "total_area_m2": 120,
  "approx_side_m": 10.95,
  "rooms": [
    {
      "name": "living",
      "area_m2": 30.0,
      "approx_width_m": 4.47,
      "approx_height_m": 6.71
    },
    // ... other rooms ...
  ]
}

4. 🧱 Material Estimation Module (material_estimator.py)
   
This module is the final stage of the AI-CPS processing pipeline. It takes the key geometric outputs—the total floor area and number of floors—and applies simplified engineering calculation rules to provide a preliminary Bill of Quantities (BoQ) for major construction materials.

⚙️ Module Functionality

The estimate_materials function calculates the requirements for three core components: Concrete, Steel, and Masonry (Bricks).

1. Concrete Volume Calculation (Slab and Structural Frame)
   
This estimation assumes a constant slab thickness and a small factor for columns and beams (structural frame):

->Slab Volume:
            $$\text{Concrete}_{\text{Slab}} = \text{Area}_{\text{m2}} \times \text{Slab Thickness}_{\text{m}} \times \text{Floors}$$

     Slab Thickness is set to $0.12 m$.

->Total Volume: The calculated slab volume is increased by a 5% factor ($1.05$) to conservatively account for the concrete needed in columns, beams, and foundation pads (assuming a basic, shallow foundation).

->Code Implementation:

concrete_volume_m3 = area_m2 * slab_thickness_m * num_floors
concrete_volume_m3 *= 1.05  # Factor for columns, beams, foundation

2. Steel (Reinforcement) Mass Calculation

Steel estimation uses a standard industry unit rate based on the total constructed floor area:

->Unit Rate: A rate of $60 \text{ kg/m}^2$ of total area is applied, which is typical for reinforced concrete structures.

->Total Steel:
$$\text{Steel}_{\text{kg}} = 60 \times \text{Area}_{\text{m2}} \times \text{Floors}$$

3. Masonry (Bricks/Blocks) Count Calculation
   
This estimation focuses on the external walls based on the building's perimeter:

->Perimeter Estimate: Assumes a square building shape for simplicity: $\text{Perimeter} \approx 4 \times \sqrt{\text{Area}_{\text{m2}}}$.

->Total Wall Area: $\text{Wall Area} = \text{Perimeter} \times \text{Wall Height}$
   -Wall Height is $3$ meters per floor.
   
->Brick Count: Calculates the number of bricks required by dividing the wall area by the standard area of one brick (assumed to be $0.075 m^2$ including mortar).

->Code Implementation (Simplified Perimeter):
perim = 4 * math.sqrt(area_m2) 
wall_area = perim * (3 * num_floors)
bricks = int(wall_area / 0.075)

⚠️ Assumptions and Limitations

The estimation provides preliminary figures for feasibility, but users should note the following simplifications:

1.Uniform Structure: Assumes a simple structural frame (slab and beam) suitable for the inputs. It does not account for complex foundation types (e.g., piles).

2.Square Geometry: The perimeter is calculated assuming a square footprint, which may overestimate/underestimate for irregular shapes (L-shape, etc.).

3.No Openings/Waste: The calculation does not deduct for windows, doors, or wall openings, nor does it factor in construction waste.

▶️ Execution Example

To test the estimation independently, run the script directly:

python src/material_estimator.py

Example Output (for $120 m^2$ and 1 floor)

{'concrete_m3': 15.12, 'steel_kg': 7200, 'bricks_count': 1394}


5. 🌐 FastAPI Deployment Module (src/main.py)

This module serves as the AI-Based Construction Planning System (AI-CPS) API Endpoint. It integrates the trained Machine Learning model, the Parametric Floorplan Generator, and the Material Estimator into a single, high-performance web service using FastAPI.

This is the central nervous system of the project, handling user input, orchestrating the predictions, and returning the complete planning package.

⚙️ Core Functionality and Architecture

The application defines a single main endpoint, /predict, which executes the entire AI-CPS pipeline sequentially:

1.Input Handling: Receives all land and project parameters via a POST request (using Form data).

2.Model Loading: Loads the saved classifier.joblib pipeline and label_encoder.joblib on startup.

3.ML Prediction: The input data is passed to the trained pipeline (model.predict(X)), which outputs a numerical label. This is immediately converted to the human-readable Predicted Building Type (e.g., "duplex", "warehouse") using the label_encoder.

4.Confidence Score: Calculates the model's confidence in its prediction using model.predict_proba(X).

5.Floorplan Generation: Calls the imported generate_floorplan function with the predicted type and area.

6.Material Estimation: Calls the imported estimate_materials function with the area and number of floors.

7.Response: Returns a single JSON response containing the prediction, confidence, floorplan breakdown, and material list.

🚀 Deployment and Usage

Prerequisites

Ensure the following components are correctly structured in your project directory:

1.The trained model files are located at: models/classifier.joblib and models/label_encoder.joblib.

2.The utility files (parameter.py and material.py) are correctly imported (as shown in the script).

1. Install Dependencies
   
Ensure your environment has the core backend and ML libraries installed:

pip install fastapi uvicorn pandas joblib xgboost scikit-learn

2. Run the API Server
   
Start the application using the ASGI server, Uvicorn (assuming the script is located in src/main.py):

uvicorn src.main:app --reload

->The application will be accessible at: http://127.0.0.1:8000.

3. Access Interactive Documentation (Swagger UI)

You can test the endpoint and view the required input parameters directly at the automatically generated Swagger UIpage:

$$\text{[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)}$$

4. Example Request Payload (API Test)
   
The endpoint expects multipart/form-data inputs (as defined by Form(...)).

Parameter Name,Data Type,Example Value,Description
area_m2,float,250.0,Land area.
slope_percent,float,2.5,Site gradient.
bearing_capacity_kpa,float,180.0,Soil strength.
project_requirement,string,"""house""",User's stated goal.
plot_shape,string,"""rectangular""",Shape of the land plot.
num_floors,int,2,Number of stories requested.
budget_usd,float,150000.0,Total budget estimate.

✅ Final Output Structure

The API will return a structured JSON response similar to the following:

{
  "Predicted Building Type": "duplex",
  "Confidence": 0.987,
  "Floorplan": {
    "template": "duplex",
    "total_area_m2": 250.0,
    "approx_side_m": 15.81,
    "rooms": [
      // ... list of rooms with areas and dimensions ...
    ]
  },
  "Materials Required": {
    "concrete_m3": 39.38,
    "steel_kg": 30000,
    "bricks_count": 3968
  }
}


🚀 Original Idea & MVP Scope (Important Note)
Original Vision

The original idea of this project was to build a fully automated AI-based construction planning system that starts with drone-based land scanning.

In the complete version, the system would:

->Capture land data using drone imagery

->Analyze terrain, plot boundaries, and elevation

->Combine drone data with soil reports

->Automatically generate construction-ready designs and estimates

->This approach would allow end-to-end automation with minimal manual input.

Current Implementation (MVP Version)

Due to limited hackathon time constraints, implementing drone data collection and image-based land analysis was not feasible.

Therefore, in this hackathon version, we adopted a practical and realistic MVP approach:

Users manually provide:

->Land measurements

->Soil bearing capacity

->Slope information

->Project requirements

The AI system then:

->Predicts the most suitable building type

->Generates a parametric floorplan

->Estimates construction materials

This allows us to demonstrate the core AI intelligence and decision-making logic while keeping the system functional and testable within the given time.

Why This Approach Was Chosen

Ensures a working end-to-end prototype

Focuses on AI-driven decision making

Reduces dependency on hardware (drones)

Allows faster validation of the core concept

Makes the system easy to extend in future versions

Future Extension

The current system is designed in a modular way, making it easy to integrate drone-based land scanning in future releases.

Planned future enhancements include:

Drone image ingestion

Terrain and boundary detection using computer vision

Automatic extraction of land dimensions

Integration of real soil test reports

Fully automated construction planning pipeline

Hackathon Relevance

This MVP demonstrates:

Clear problem understanding

Scalable system design

Practical AI implementation

Strong foundation for a real-world product


Planned Tools for Full-Scale Implementation

The original vision of the project involves automated land data extraction using drone imagery.
Due to time constraints, these tools were not implemented in the hackathon MVP but are planned for future versions.

🔍 Computer Vision & Image Processing

OpenCV

Land boundary detection

Plot shape extraction

Terrain analysis

Elevation and slope estimation

NumPy

Image matrix processing

Feature extraction support

🛰 Drone & Geospatial Data Processing

Drone imagery / aerial photography

GeoPandas

Geospatial land data handling

Rasterio

Elevation and terrain data processing

🧠 Generative AI Enhancements

Generative models

Automatic blueprint generation

AI-assisted design variations

Text-based GenAI

Construction explanation

Design justification

User-friendly AI suggestions

🧱 Construction & Design Extensions

SVG / DXF generation

Exportable blueprint drawings

3D Visualization Libraries

Room-based 3D model previews

BIM Integration (Future Scope)

Why These Tools Matter

Using tools like OpenCV and drone imagery will allow the system to:

Eliminate manual measurement input

Automatically extract land dimensions

Improve accuracy of planning

Provide a fully automated end-to-end solution

The current MVP is intentionally designed to be modular, making it easy to integrate these tools without major changes to the architecture.

Hackathon MVP Justification

Due to limited development time:

Drone data capture

Image-based land analysis

OpenCV pipelines

were deferred to future versions.

Instead, the focus was on validating:

AI decision-making

Model accuracy

End-to-end system integration

This ensures a working prototype with a clear roadmap.

