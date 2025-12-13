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
