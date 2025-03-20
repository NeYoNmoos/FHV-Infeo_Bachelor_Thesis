from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon
from sklearn.neighbors import NearestNeighbors
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# Load preprocessed tracking data
track_features = pd.read_parquet("../Data_Preperation/osm_style_data/gps_data_osm_style.parquet")

# Normalize features for Nearest Neighbors
scaler = StandardScaler()
features_to_scale = track_features.drop(columns=["id_tracking"], errors="ignore")
scaler.fit(features_to_scale)

# Print column types to verify only numeric ones remain
print("Features to scale:")
print(features_to_scale.dtypes)
print("Tracking data feature names:", features_to_scale.columns.tolist())

# Apply StandardScaler
track_features_scaled = scaler.transform(features_to_scale)

# Train Nearest Neighbors model
nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(track_features_scaled)

# Initialize FastAPI
app = FastAPI()

# API Input Model
class GeofenceRequest(BaseModel):
    geofence: List[List[float]]  # List of [lat, lon] pairs defining the polygon


def extract_osm_features(geofence_poly):
    """
    Extracts OSM-style road network features from the given geofence.
    """
    try:
        # Convert geofence to bounding box
        min_lon, min_lat, max_lon, max_lat = geofence_poly.bounds

        # Get the road network from OSM
        G = ox.graph_from_bbox((min_lon, min_lat, max_lon, max_lat), network_type="drive")

        # Compute the area of the geofence in km² using geopandas
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(geometry=[geofence_poly], crs="EPSG:4326").to_crs(epsg=3857)
        area_km2 = gdf.geometry.area.iloc[0] / 1e6

        # Compute OSM-based road statistics
        stats = ox.basic_stats(G, area=area_km2)

        # Extract and align with tracking data
        osm_features = {
            "total_distance": stats.get("edge_length_total", 0) / 1000,  # Convert meters to km
            "node_density": stats.get("node_density_km", 0),
            "street_density": stats.get("street_density_km", 0),
            "avg_street_segment_length": stats.get("street_length_avg", 0),
            "intersection_density": stats.get("intersection_density_km", 0),
            "circuity_avg": stats.get("circuity_avg", 0),
            "self_loop_proportion": stats.get("self_loop_proportion", 0),
            "street_segment_count": stats.get("street_segment_count", 0),
            "streets_per_node_avg": stats.get("streets_per_node_avg", 0)
        }

        # Convert to numpy array for nearest neighbor matching
        return np.array([osm_features[key] for key in osm_features.keys()]).reshape(1, -1)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OSM feature extraction error: {e}")



@app.post("/suggest_tracking/")
def suggest_tracking(geo_request: GeofenceRequest):
    try:
        # Convert geofence to Polygon
        geofence_poly = Polygon(geo_request.geofence)

        # Extract road network features from OpenStreetMap
        osm_feature_vector = extract_osm_features(geofence_poly)

        # Normalize OSM feature vector using the same scaler as GPS tracking features
        osm_feature_vector_scaled = scaler.transform(osm_feature_vector)

        # Find closest historical track
        distances, indices = nn.kneighbors(osm_feature_vector_scaled)
        
        print("distances: ", distances, "indices: ", indices)
        closest_routes = track_features.iloc[indices[0]]

        # Convert results to JSON
        result = []
        for _, row in closest_routes.iterrows():
            result.append({
                "id_tracking": int(row["id_tracking"]),
                "total_distance_km": round(row["total_distance"], 2),
                "node_density": round(row["node_density"], 2),
                "street_density": round(row["street_density"], 2),
                "avg_street_segment_length": round(row["avg_street_segment_length"], 2),
                "intersection_density": round(row["intersection_density"], 2),
                "circuity_avg": round(row["circuity_avg"], 2),
                "self_loop_proportion": round(row["self_loop_proportion"], 2),
                "street_segment_count": int(row["street_segment_count"]),
                "streets_per_node_avg": round(row["streets_per_node_avg"], 2)
            })


        return {"suggested_routes": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
