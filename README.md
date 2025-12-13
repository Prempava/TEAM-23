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

Next Step: Once the synthetic_data.csv file is generated, the next stage is Data Preprocessing 
and Model Training using the XGBoost classifier.

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

