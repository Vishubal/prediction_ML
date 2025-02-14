from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
#from pyngrok import ngrok
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pickle 
import os


# Initialize FastAPI app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.abspath("./src/prediction_models/car_purchase_decision_model.pkl")
scaler_path = os.path.abspath("./src/prediction_models/car_purchase_decision_scaler.pkl")

loan_model_path = os.path.abspath("./src/prediction_models/loan_prediction_model.pkl")
loan_scaler_path = os.path.abspath("./src/prediction_models/loan_prediction_scaler.pkl")


# Load the saved model and scaler
# car_model = pickle.load(open('./src/prediction_models/car_purchase_decision_model.pkl', 'rb'))
# car_scaler = pickle.load(open('./src/prediction_models/car_purchase_decision_scaler.pkl', 'rb'))
# Load the saved model and scaler
try:
    with open(model_path, "rb") as model_file:
        car_model = pickle.load(model_file)

    with open(scaler_path, "rb") as scaler_file:
        car_scaler = pickle.load(scaler_file)

    with open(loan_model_path, "rb") as loan_model_file:
        loan_model = pickle.load(loan_model_file)

    with open(loan_scaler_path, "rb") as loan_scaler_file:
        loan_scaler = pickle.load(loan_scaler_file)

    print("âœ… Model and Scaler Loaded Successfully!")
except Exception as e:
    print(f"âŒ Error Loading Model: {e}")


# # Define input schema
class CarPurchase(BaseModel):
  features: list

class LoanEligibility(BaseModel):
  features: list

# Welcome Endpoint
@app.get("/")
def read_root():
  return {"message": "Welcome to the Model Prediction API! ✔"}

# Define prediction endpoint
@app.post("/predict/car_purchase")
def predict_car_purchase(input_data: CarPurchase):
  # Convert input list to NumPy array
  input_array = np.array(input_data.features).reshape(1, -1)

  # scale the input features
  scaled_data = car_scaler.transform(input_array)

  # Make prediction
  prediction = car_model.predict(scaled_data)
  result = "Car purchase approved" if prediction[0] == 1 else "Car purchase denied"

  return {"prediction": result}

# Define loan eligibility prediction endpoint
@app.post("/predict/loan_eligibility")
def predict_loan_eligibility(input_data: LoanEligibility):
  # Convert input list to NumPy array
  input_array = np.array(input_data.features).reshape(1, -1)

  # scale the input features
  scaled_data = loan_scaler.transform(input_array)

  # Make prediction
  prediction = loan_model.predict(scaled_data)
  result = "Loan approved" if prediction[0] == 1 else "Loan declined"

  return {"prediction": result}

# Run the server
# nest_asyncio.apply()

#uvicorn.run(app, host="0.0.0.0", port=8000)


