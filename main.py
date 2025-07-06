from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from preprocessor import PreprocessedText

# Load the trained model
with open('trained_pipeline-0.1.0.sav', 'rb') as f:
    model = pickle.load(f)


class EmailRequest(BaseModel):
    subject: str
    body: str


app = FastAPI()

@app.get("/")
def root():
    return {"message": "Go to /docs"}

@app.post("/predict")
def predict_spam(email: EmailRequest):
    full_text = f"Subject: {email.subject} {email.body}"
    
    prediction = model.predict([full_text])[0]
    
    return {"prediction": prediction}
