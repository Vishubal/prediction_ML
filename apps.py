from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
#from pyngrok import ngrok
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pickle 


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

# Load the saved model and scaler
loan_prediction_model = pickle.load(open('./src/prediction_models/loan_prediction_decision_model.pkl', 'rb'))
loan_prediction_scaler = pickle.load(open('./src/prediction_models/loan_predictione_decision_scaler.pkl', 'rb'))


# Define input schema
class LoanPrediction(BaseModel):
  features: list

# Welcome Endpoint
@app.get("/")
def read_root():
  return {"message": "Welcome to the Model Prediction API! ✔"}

# Define prediction endpoint
@app.post("/predict/loan_prediction")
def predict_loan_eligibility(input_data: LoanPrediction):
  # Convert input list to NumPy array
  input_array = np.array(input_data.features).reshape(1, -1)

  # scale the input features
  scaled_data = loan_prediction_scaler.transform(input_array)

  # Make prediction
  prediction = loan_prediction_model.predict(scaled_data)
  result = "Loan approved" if prediction[0] == 1 else "Loan denied"

  return {"prediction": result}

# Run the server
nest_asyncio.apply()

#uvicorn.run(app, host="0.0.0.0", port=8000)


