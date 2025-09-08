from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

class PlayerData(BaseModel):
    Age: int
    Height_cm: int
    Weight_kg: int
    Training_Hours_Per_Week: float
    Matches_Played_Past_Season: int
    Previous_Injury_Count: int
    Knee_Strength_Score: float
    Hamstring_Flexibility: float
    Reaction_Time_ms: float
    Balance_Test_Score: float
    Sprint_Speed_10m_s: float
    Agility_Score: float
    Sleep_Hours_Per_Night: float
    Stress_Level_Score: float
    Nutrition_Quality_Score: float
    Warmup_Routine_Adherence: int
    Injury_Next_Season: int
    BMI: float
    Position_Defender: bool
    Position_Forward: bool
    Position_Goalkeeper: bool
    Position_Midfielder: bool

    class Config:
        schema_extra = {
            "example": {
                "Age": 24,
                "Height_cm": 182,
                "Weight_kg": 75,
                "Training_Hours_Per_Week": 5.49,
                "Matches_Played_Past_Season": 17,
                "Previous_Injury_Count": 3,
                "Knee_Strength_Score": 81.16,
                "Hamstring_Flexibility": 83.48,
                "Reaction_Time_ms": 264.19,
                "Balance_Test_Score": 84.0,
                "Sprint_Speed_10m_s": 3.5,
                "Agility_Score": 69.0,
                "Sleep_Hours_Per_Night": 7,
                "Stress_Level_Score": 78.0,
                "Nutrition_Quality_Score": 74.0,
                "Warmup_Routine_Adherence": 1,
                "Injury_Next_Season": 0,
                "BMI": 22.6,
                "Position_Defender": False,
                "Position_Forward": True,
                "Position_Goalkeeper": False,
                "Position_Midfielder": False
            }
        }
        
app = FastAPI(
    title="Football Player Injury Predictor",
    description= "Predicts whether a football player will have an injury in the next season based on their current stats",
    version="1.0.0"
)

model_path = os.path.join("models", "footballmod.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
@app.post("/predict")
def predict_injury_score(player: PlayerData):
    """
    Predict injury
    """
    
    features = np.array([[
        player.Age, 
        player.Height_cm,
        player.Weight_kg,
        player.Training_Hours_Per_Week,
        player.Matches_Played_Past_Season,
        player.Previous_Injury_Count,
        player.Knee_Strength_Score,
        player.Hamstring_Flexibility,
        player.Reaction_Time_ms,
        player.Balance_Test_Score,
        player.Sprint_Speed_10m_s,
        player.Agility_Score,
        player.Sleep_Hours_Per_Night,
        player.Stress_Level_Score,
        player.Nutrition_Quality_Score,
        player.Warmup_Routine_Adherence,
        player.Injury_Next_Season,
        player.BMI,
        player.Position_Defender,
        player.Position_Forward,
        player.Position_Goalkeeper,
        player.Position_Midfielder
    ]])
    
    prediction = model.predict(features)[0]
    
    return {
        "predicted_injury_score": round(prediction, 2),
        "interpretation": get_interpretation(prediction)
    }

def get_interpretation(score):
    """Provide human-readable interpretation of score"""
    if score == 1:
        return "Player will be injured in the upcoming season"
    else:
        return "Player will not be injured in the upcoming season"
    
@app.get("/")
def health_check():
    return{"status": "healthy", "model": "football_injury_v1"}
