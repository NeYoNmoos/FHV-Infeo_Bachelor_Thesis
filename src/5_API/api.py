from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from src.utilities.feature_extraction import preprocess_tracking_data
from src.AWM_API.api_calls import get_access_token, get_single_tracking

app = FastAPI()
clf = joblib.load("src/4_Train_Classifier/rf_area_classifier_full.pkl")


@app.get("/classify/{tracking_id}")
async def classify_tracking(tracking_id: str):
    try:
        access_token = await get_access_token()
        if not access_token:
            raise HTTPException(status_code=401, detail="Failed to retrieve access token.")

        raw_tracking = await get_single_tracking(access_token, tracking_id)

        features_df = preprocess_tracking_data(raw_tracking)
        if features_df.empty:
            raise HTTPException(status_code=400, detail="Preprocessing failed or returned empty features.")

        features_cleaned_df = features_df.drop(columns=["num_stops", "total_distance"])
        prediction = clf.predict(features_cleaned_df)[0]
        

        return JSONResponse(content={
            "tracking_id": tracking_id,
            "predicted_class": prediction
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
