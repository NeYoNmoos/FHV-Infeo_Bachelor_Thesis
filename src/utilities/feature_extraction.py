import pandas as pd
import numpy as np
from geopy.distance import geodesic
from math import radians, sin, cos, atan2, pi

def calculate_bearing(p1, p2):
    lat1, lon1 = radians(p1[0]), radians(p1[1])
    lat2, lon2 = radians(p2[0]), radians(p2[1])
    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    return bearing

def preprocess_tracking_data(tracking):
    try:
        points = tracking.get("points", [])
        if not points or len(points) < 2:
            raise ValueError("Not enough GPS points for feature extraction.")

        df = pd.DataFrame(points)
        df = df.dropna(subset=["latitude", "longitude"])
        df = df.sort_values(by="sequence")

        coords = list(zip(df["latitude"], df["longitude"]))
        if len(coords) < 2:
            raise ValueError("Not enough valid coordinates.")

        num_points = len(coords)

        lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
        lon_min, lon_max = df["longitude"].min(), df["longitude"].max()

        height = geodesic((lat_min, lon_min), (lat_max, lon_min)).meters
        width = geodesic((lat_min, lon_min), (lat_min, lon_max)).meters
        bbox_area = height * width

        point_density = num_points / (bbox_area + 1e-6)
        dists = [geodesic(coords[i], coords[i+1]).meters for i in range(len(coords)-1)]
        avg_segment_distance = np.mean(dists)
        total_distance = np.sum(dists)

        num_stops = (df.get("speed", pd.Series([])) == 0).sum()
        duration = tracking.get("duration", 0)
        length = tracking.get("length", 0)

        straight_line = geodesic(coords[0], coords[-1]).meters
        straightness = straight_line / total_distance if total_distance > 0 else 0

        angle_changes = []
        for i in range(1, len(coords) - 1):
            prev_bearing = calculate_bearing(coords[i - 1], coords[i])
            next_bearing = calculate_bearing(coords[i], coords[i + 1])
            delta = abs(next_bearing - prev_bearing)
            if delta > pi:
                delta = 2 * pi - delta
            angle_changes.append(delta)
        mean_heading_change = np.mean(angle_changes) if angle_changes else 0

        features = {
            "num_points": num_points,
            "bbox_area": bbox_area,
            "point_density": point_density,
            "avg_segment_distance": avg_segment_distance,
            "total_distance": total_distance,
            "straightness": straightness,
            "mean_heading_change": mean_heading_change,
            "num_stops": num_stops,
            "duration": duration,
            "length": length
        }

        return pd.DataFrame([features])

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")
