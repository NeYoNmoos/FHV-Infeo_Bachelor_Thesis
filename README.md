# Classification of GPS Track Data Using AI Methods: A Case Study of Waste Collection Vehicles

This repository contains the full implementation and documentation for my Bachelor Thesis:

**"Classification of GPS Track Data Using Machine Learning Methods: A Case Study of Waste Collection Vehicles"**

## Abstract

In waste management, strategic route planning is a central
process in which an efficient allocation of areas is intended to achieve
optimal
utilization of the vehicle fleet while minimizing costs. However, waste
management companies are often faced with uncertainties, especially when
planning operations in new, unknown areas, as
experience is lacking and many planning-relevant assumptions have to be
estimated.

This work investigates how existing GPS tracking data from
waste collection vehicles can be used to automatically classify the structural
characteristics of
areas. The aim is to extract
patterns from known areas that allow conclusions to be drawn about the spatial
structure (e.g.
urban vs. rural) and can thus serve as a basis for comparisons with new
areas.

To this end, a machine learning-based system was developed that processes
GPS waypoint data, extracts features, structures the data using a
semi-supervised clustering method and then trains a
classification model that automatically classifies new data sets. The
classification is carried out in four structure classes: URBAN,
SUBURBAN, TOWN and RURAL.

The classification model was embedded in an API that enables a simple
connection to existing software solutions. Automated classification can be
carried out by specifying a
tracking ID.

In the long term, this work thus forms the basis for an expandable
solution in which geographical structural data (e.g. from OpenStreetMap) is
also included in
the analysis in order to create robust comparison options between areas known
to
and new areas. However, the complete integration of these
additional data sources is not part of this bachelor project,
but is reserved for future work.

## Technologies Used

- **Python** (Jupyter Notebooks)
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- `geopy`, `pyarrow`, `sqlalchemy`
- `SHAP` for model explainability
- `FastAPI` for prototype REST API
- LaTeX for documentation

## Core Concepts

- GPS data preprocessing and outlier filtering (DBSCAN)
- Feature extraction (bounding box, point density, heading changes, etc.)
- Feature weighting using labeled samples
- Semi-supervised clustering with weighted K-Means and custom centroids
- Random Forest classifier with performance evaluation
- SHAP explainability
- PCA and map visualizations

## Project Structure

```
├── src/
│ └── 1_Data_Analysis/
│ └── 2_Feature_Extraction/
│ └── 3_Analysis_on_Extracted_Data/
│ └── 4_Train_Classifier/
│ └── Data_Preperation/
│ └── utilities/
├── Thesis/
│ └── Bachelor_Thesis_GPS_Classification.pdf
│ └── Bachelor_Thesis_GPS_Classification.tex (main thesis document)
└── README.md
```

## Results

- **Classifier Accuracy:** 98.3%
- **Macro F1-Score:** 98.1%
- **Silhouette Score (Clustering):** 0.26
- **SHAP Insights:** Key features such as `bbox_area`, `point_density`, and `avg_segment_distance` show strong monotonic behavior across urban categories.

## Future Work

- Integration with OpenStreetMap (OSM) for structural context
- Filtering non-operational transit segments
- Expanded sampling for better clustering seeds
- Deployment as an API for live use by municipalities or route planning software

## Thesis Document

You can find the full thesis PDF in the `Thesis/` folder. All formulas, figures, and evaluations are included and explained in detail.

## Acknowledgements

Special thanks to **infeo GmbH** for providing the dataset and insights into real-world waste collection logistics.
And to **Dipl.-Ing. Dr. techn. Ralph Hoch** for providing guidance throughout the development of this thesis.
