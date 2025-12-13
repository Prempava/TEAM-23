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
