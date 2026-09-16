from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
import pandas as pd
from tensorflow.keras.models import load_model
import joblib


templates=Jinja2Templates(directory='templates')

app=FastAPI()

model=load_model('model.keras')
scaler=joblib.load('scaler.pkl')

@app.get("/")
def index(request:Request):
    return templates.TemplateResponse(request=request,name="index.html")


@app.post("/predict")
def predict(request:Request,
        gre:int=Form(...),
        toefl:int=Form(...),
        university_rating:int=Form(...),
        sop:float=Form(...),
        lor:float=Form(...),
        cgpa:float=Form(...),
        research:int=Form(...)


        ):

    new_df = pd.DataFrame({
        'gre score': [gre],
        'toefl score': [toefl],
        'university rating': [university_rating],
        'sop': [sop],
        'lor': [lor],
        'cgpa': [cgpa],
        'research': [research]
    })

    new_df_scaled=scaler.transform(new_df)

    new_df_pred=model.predict(new_df_scaled)

    print("prediction: ",new_df_pred)

    return templates.TemplateResponse(
        name='index.html',
        request=request,
        context={
            "prediction":new_df_pred
        }
    )