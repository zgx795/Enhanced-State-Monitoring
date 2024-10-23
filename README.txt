
This project focuses on the preparation, processing, modeling, and evaluation of a dataset collected over a specified period. The data undergoes several stages, including preprocessing, feature selection, model building, and transfer learning. The project culminates in health evaluation using an improved SVDD model.

 1. Data Preparation
The dataset was collected based on empirical observations and literature over a period, ensuring appropriate intervals and sufficient data quantity.

 2. Data Processing
Missing Data Handling: Missing values are handled using interpolation.
VMD Filtering: Variational Mode Decomposition (VMD) is applied for further data cleaning.
Feature Selection: Random Forest algorithm is used for feature selection, ensuring the selection of highly correlated features.

 3. Model Building
A multi-input, multi-output model is built using the **BO-CNN-BiLSTM** approach. The model is then compared with other models for analysis and evaluation on the test set.

Model Script: `train_model.m`
Model Output Comparison: The output from the original model is analyzed and compared using the script `cpmd.m`, with the following support:
 Radar Chart Subprogram: `radarChart.m`
 Pre-trained Models: Provided as `.mat` files

 4. Model Self-Update (Increment Learning)
The project implements a model self-updating mechanism using transfer learning. A new local historical dataset is utilized to fine-tune the existing large-scale model, allowing the model to adapt over time.

Transfer Learning Script: `finetunemd.m` (For model fine-tuning)
New vs Old Model Comparison: Compare predictions from new and old models on both new and old test datasets.
Execution: Use the `main` function to run `train_model.m` and `finetunemd.m` for the required steps.

 5. Monitoring and Evaluation
System health evaluation is performed using an improved SVDD (Support Vector Data Description) model, which assesses the system's condition and verifies the model’s effectiveness.

SVDD Model Script: `TrainSVDD` (For health evaluation)
Test Script: `TestSVDD` (For evaluating trained models)
