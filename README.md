# GPS-Based Urban Structure Classification for Waste Collection

This repository contains the full implementation and documentation for my Bachelor Thesis:

**"Classification of Urban Structures Based on GPS Tracking Data of Waste Collection Vehicles"**

## Abstract

In waste management, strategic route planning is a crucial process where
optimal fleet utilization is determined through the efficient division of
service areas, with the goal of minimizing costs. This process is applied by
waste disposal companies both for existing service areas and when calculating
bids for new tenders. Especially in regions where there is no prior experience,
numerous uncertain assumptions and estimates must be made for robust route
planning. To reduce these uncertainties through the analysis of geographical
structures, a technology will be integrated into the company’s existing route
planning software, which can automatically solve the following task: Based on
existing GPS records, the structural characteristics of the respective
collection areas should be numerically evaluated and classified. Additionally,
geographical structural data (preferably from freely available sources) from
unknown areas should be collected and classified in the same way. This approach
will create both a reference data set (from existing collection routes) and a
comparison data set (from new tender areas). Where the classification data
match, it can be assumed that planning-relevant parameters from existing
service areas can be applied to the new areas without risky assumptions. The
classification of GPS data and geographical structural data should be automated
using artificial intelligence. Furthermore, the consideration of which
geographical structural data are meaningful for comparison should, if
necessary, also be supported by AI technologies.

The practical goal of this work is to implement a sandbox service that can be
called and populated with data by the existing software of infeo GmbH,
enabling the
creation of classifications and comparisons of GPS data and tender structural
data "at the push of a button." This will provide users with the ability to
calculate appropriate planning parameters from their existing service areas for
new tenders, thereby significantly reducing uncertainties in bid calculations.

## Technologies Used

- **Python** (Jupyter Notebooks)
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- `geopy`, `pyarrow`, `sqlalchemy`
- `SHAP` for model explainability
- `Flask` for prototype REST API
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

bachelor_thesis_infeo
├── src
│ └── 1_Data_Analysis
│ └── 2_Feature_Extraction
│ └── 3_Analysis_on_Extracted_Data
│ └── 4_Train_Classifier
├── Thesis
├── Thesis
│ └── Bachelor_Thesis_GPS_Classification.tex (main thesis document)
└── 📄 README.md

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

## License

This work is provided under the MIT License. See `LICENSE` for details.

## Acknowledgements

Special thanks to **infeo GmbH** for providing the dataset and insights into real-world waste collection logistics.

---

If you use or build upon this work, please consider citing the thesis or giving credit.
