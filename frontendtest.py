import gradio as gr
import requests

fastapiurl = "http://3.148.227.228:8000/predict"

def call_fastapi(
    Age, 
    Height_cm, 
    Weight_kg, 
    Training_Hours_Per_Week, 
    Matches_Played_Past_Season,
    Previous_Injury_Count,
    Knee_Strength_Score,
    Hamstring_Flexibility,
    Reaction_Time_ms,
    Balance_Test_Score,
    Sprint_Speed_10m_s,
    Agility_Score,
    Sleep_Hours_Per_Night,
    Stress_Level_Score,
    Nutrition_Quality_Score,
    Warmup_Routine_Adherence,
    BMI,
    Position_Defender,
    Position_Forward,
    Position_Goalkeeper,
    Position_Midfielder
):
    playerdata = {
        "Age": int(Age),
        "Height_cm": int(Height_cm),
        "Weight_kg": int(Weight_kg),
        "Training_Hours_Per_Week": Training_Hours_Per_Week,
        "Matches_Played_Past_Season": int(Matches_Played_Past_Season),
        "Previous_Injury_Count": int(Previous_Injury_Count),
        "Knee_Strength_Score": Knee_Strength_Score,
        "Hamstring_Flexibility": Hamstring_Flexibility,
        "Reaction_Time_ms": Reaction_Time_ms,
        "Balance_Test_Score": Balance_Test_Score,
        "Sprint_Speed_10m_s": Sprint_Speed_10m_s,
        "Agility_Score": Agility_Score,
        "Sleep_Hours_Per_Night": Sleep_Hours_Per_Night,
        "Stress_Level_Score": Stress_Level_Score,
        "Nutrition_Quality_Score": Nutrition_Quality_Score,
        "Warmup_Routine_Adherence": int(Warmup_Routine_Adherence),
        "BMI": BMI,
        "Position_Defender": Position_Defender,
        "Position_Forward": Position_Forward,
        "Position_Goalkeeper": Position_Goalkeeper,
        "Position_Midfielder": Position_Midfielder
    }
    try:
        response = requests.post(fastapiurl, json=playerdata)
        response.raise_for_status()
        return response.json()  # your FastAPI should return {"prediction": ...}
    except Exception as e:
        return {"error": str(e)}
    
interface = gr.Interface(
    fn = call_fastapi, 
    inputs=[
        gr.Slider(0, 100, label="Age"),
        gr.Slider(0, 350, label="Height_cm"),
        gr.Slider(0, 10, label="Weight_kg"),
        gr.Slider(0, 45, label="Training_Hours_Per_Week"),
        gr.Slider(0, 45, label="Matches_Played_Past_Season"),
        gr.Slider(0, 10, label="Previous_Injury_Count"),
        gr.Slider(0, 150, label="Knee_Strength_Scorem"),
        gr.Slider(0, 150, label="Hamstring_Flexibility"),
        gr.Slider(0, 500, label="Reaction_Time_ms"),
        gr.Slider(0, 84, label="Balance_Test_Score"),
        gr.Slider(0, 3.5, label="Sprint_Speed_10m_s"),
        gr.Slider(0, 150, label="Agility_Score"),
        gr.Slider(0, 14, label="Sleep_Hours_Per_Night"),
        gr.Slider(0, 150, label="Stress_Level_Score"),
        gr.Slider(0, 150, label="Nutrition_Quality_Score"),
        gr.Slider(0, 1, label="Warmup_Routine_Adherence"),
        gr.Slider(0, 100, label="BMI"),
        gr.Slider(0, 1, label="Position_Defender"),
        gr.Slider(0, 1, label="Position_Forward"),
        gr.Slider(0, 1, label="Position_Goalkeeper"),
        gr.Slider(0, 1, label="Position_Midfielder"),
    ],
    outputs = "json",
    title="Soccer Injury Predictor",
    description = "This app uses a logistic regression model to determine whether a soccer player will have an injury in the upcoming season or not."
    )   

interface.launch(share=True)
