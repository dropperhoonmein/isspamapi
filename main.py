from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from preprocessor import PreprocessedText
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
with open('trained_pipeline-0.1.0.sav', 'rb') as f:
    model = pickle.load(f)


class EmailRequest(BaseModel):
    subject: str
    body: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Go to /docs"}

@app.post("/predict")
def predict_spam(email: EmailRequest):
    full_text = f"Subject: {email.subject} {email.body}"
    probability = model.predict_proba([full_text])[0].tolist()
    
    return {"prediction": prediction,"probability":probability}
