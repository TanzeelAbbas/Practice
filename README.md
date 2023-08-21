# SkyElectric Coding Task for ML Interns

## Introduction
This project shows how to build a Machine Learning (ML) model that predicts a surface area based on its radius (r) and height (h). The ML model is trained on a dataset of input-output pairs, and it makes use of the Random Forest and Neural Network techniques.

## Dataset
The dataset used in this project contains randomly generated values for radius and height, along with their corresponding surface area values calculated using the formula `A = 2πrh + 2πr^2`. The dataset is split into training and testing sets for model development and evaluation.

### Dataset Files
`train_data.csv`: Training dataset containing input features (radius and height) and traget surface area.
`test_data.csv`: Testing dataset for model evaluation.

## Machine Learning Models

### Neural Network Model
A feedforward Neural Network was implemented using TensorFlow and Keras. The neural network consists of multiple dense layers and was trained to predict the surface area. MSE was used as the loss function for training.

### Random Forest Model
We employed the Random Forest Regressor from Scikit-learn to create a Random Forest model for predicting the surface area. The model was trained on the training dataset and evaluated using Mean Squared Error (MSE) on the testing dataset.

## Usage
To use the trained models for inference, you can run the provided inference script. You will need to have Python and the required libraries installed. The script takes radius and height as input and outputs the predicted surface area.

### Inference Script
`inference_script.py`: Use this script to make predictions with the trained models. Adjust the input values (radius and height) to obtain predictions for different surface area configurations.

## Dependencies
Python (>= 3.6)
NumPy
Pandas
pickle
Scikit-learn
TensorFlow (>=2.0)
joblib

## Files
Train_Data.csv: Training dataset
Test_Data.csv: Testing dataset
Random_Forest_Model.pkl: Trained Random Forest model
Neural_Network_Model.h5: Trained Neural Network model
Neural Network Implementation Model Code.ipynb: Script for implement Neural Network 
Random Forest Implementation Model Code.ipynb: Script for Implement Random Forest 
Inference script for Random Forest Model.ipynb: Script for making predictions from Random Forest Model
Inference Script for Neural Network Model.ipynb: Script for making predictions from Neural Network Model

## Conclusion
This project demonstrates the use of ML models to predict the surface area of a cylinder based on its dimensions. Both the Random Forest and Neural Network models were trained and evaluated, allowing for accurate predictions.

### Name: Tanzeel Abbas
### Email: tanzeelabbas114@gmail.com
