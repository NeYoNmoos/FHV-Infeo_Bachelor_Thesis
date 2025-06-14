from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from shapely.geometry import shape
import osmnx as ox
from osmnx.projection import project_geometry
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import uvicorn
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

# Load clustered and cleaned OSM dataset
df = pd.read_csv("../3_Train_Classifier/osm_clustered_cleaned.csv")
feature_columns = [
    "street_length_total", "intersection_count", "street_density_km",
    "edge_density_km", "circuity_avg", "intersection_density_km", "node_density_km"
]
X = df[feature_columns].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(X_scaled)

# --- Input schema ---
class GeoQuery(BaseModel):
    polygon: dict  # GeoJSON-like dict (coordinates + type)

# --- Helper: extract OSM stats for geofence polygon ---
def extract_osm_features(geojson_poly: dict):
    try:
        poly = shape(geojson_poly)
        G = ox.graph_from_polygon(poly, network_type="drive")

        projected_poly, _ = project_geometry(poly)
        area = projected_poly.area
        stats = ox.basic_stats(G, area=area)
        print(stats)

        return np.array([
            stats.get("street_length_total", 0),
            stats.get("intersection_count", 0),
            stats.get("street_density_km", 0),
            stats.get("edge_density_km", 0),
            stats.get("circuity_avg", 0),
            stats.get("intersection_density_km", 0),
            stats.get("node_density_km", 0)
        ]).reshape(1, -1)

    except Exception as e:
        print("OSM extraction error:", e)
        raise HTTPException(status_code=400, detail="Failed to process geofence")

# --- API Endpoint ---
@app.post("/match_routes")
def match_routes(query: GeoQuery, top_k: int = 5):
    query_vector = extract_osm_features(query.polygon)
    query_scaled = scaler.transform(query_vector)
    print("Query Vector: ", query_vector)
    print("Query Vector: ", query_scaled)


    # similarities = cosine_similarity(query_scaled, X_scaled)[0]
    # top_indices = similarities.argsort()[::-1][:top_k]

    distances, indices = knn.kneighbors(query_scaled)

    results = df.iloc[indices[0]][["tracking_id", "cluster"]].copy()

    results = df.iloc[indices[0]][["tracking_id", "cluster"]].copy()
    results["distance_score"] = distances[0].round(4)

    return results.to_dict(orient="records")

# --- Run with: uvicorn app:app --reload ---
