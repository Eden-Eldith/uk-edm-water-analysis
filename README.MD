# UK EDM-Water Analysis

This repository contains a forensic analysis pipeline for correlating UK **Event Duration Monitoring (EDM)** sewage overflow data with **Water Quality** measurements (2020–2024). The analysis identifies high-risk pollution sites and potential marine mortality events through geospatial correlation and composite risk scoring.


# TL;DR:
 Identifies 831,706 UK sewage-linked pollution events (2020–2024) using custom Python tools and geospatial risk scoring.
---

## 🔍 Project Overview

This analysis:
- Processes 2.5GB+ of water quality CSV data and 11MB+ of EDM Excel files
- Reconciles incompatible coordinate systems (OSGR ↔ Eastings/Northings)
- Applies geospatial matching (KDTree algorithm within 5km radius)
- Calculates composite environmental risk scores based on:
  - Overflow magnitude (spill count & duration)
  - Pollution severity (dissolved oxygen, sewage indicators, etc.)
  - Proximity to water sampling points
  - Persistence across multiple years

**Key Findings:**
- 831,706 pollution correlations identified
- 1,069 overflow sites analyzed
- 29 sites classified as "Critical Risk"

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages:
  ```bash
  pip install pandas scipy numpy openpyxl
  ```

### Step 1: Download the Official Datasets

1. **EDM (Event Duration Monitoring) Data**
   - Go to: https://environment.data.gov.uk/dataset/21e15f12-0df8-4bfc-b763-45226c16a8ac
   - Download ALL Excel files for years 2020-2024
   - Files will be named like:
     - `EAv EDM Return [Company] Annual 2020.xlsx`
     - `EDM 2021 Storm Overflow Annual Return - all water and sewerage companies.xlsx`
     - etc.

2. **Water Quality Data**
   - Go to: https://environment.data.gov.uk/water-quality/view/download/new
   - Click "Download water quality data"
   - Select "All substances" (or at minimum: Ammonia, BOD, Dissolved Oxygen, Oil/Grease, Suspended Solids)
   - Download data for years 2020-2024 as CSV files
   - You'll get files named by year (e.g., `2020.csv`, `2021.csv`, etc.)

### Step 2: Organize Your Directory Structure

Create the following directory structure in your project folder:

```
uk-edm-water-analysis/
├── xlsx/          # Place all EDM Excel files here
├── csvs/          # Place all water quality CSV files here
├── batch_filter_final.py
├── analyze_correlations_final.py
└── README.md
```

**Important:** 
- Put ALL downloaded Excel files in the `xlsx/` folder
- Put ALL downloaded CSV files in the `csvs/` folder
- Keep the original filenames - the scripts parse years from them

### Step 3: Run the Analysis Pipeline

Open a terminal in the project directory and run:

```bash
# Step 1: Correlate EDM overflows with water quality samples
# This will take several minutes depending on your system
python batch_filter_final.py

# Step 2: Analyze correlations and generate risk scores
python analyze_correlations_final.py
```

### Step 4: View Results

After running both scripts, you'll have:
- `correlated_events_final.csv` - All 831,706+ correlated pollution events
- `suspicious_overflow_summary.csv` - Ranked list of overflow sites by risk score

The console output will also show:
- Top 50 most suspicious overflow sites
- Critical alerts for high-risk sites
- Overall statistics and recommendations

---

## 📊 Understanding the Output

### Risk Score Components

The composite risk score (0-100 scale) weighs:
- **35% Magnitude**: Total spill count and duration
- **30% Severity**: Type of pollution (critical DO, sewage, ammonia, etc.)
- **20% Pollution**: Measured pollutant concentrations
- **15% Proximity**: Distance to water sampling points
- **+10% bonus**: For persistent multi-year offenders

### Risk Categories
- **Critical** (75-100): Immediate investigation required
- **Very High** (50-75): Urgent investigation recommended  
- **High** (25-50): Priority monitoring needed
- **Moderate** (10-25): Standard monitoring
- **Low** (0-10): Minimal immediate concern

### Key Pollutant Types
- **Dissolved Oxygen (DO)**: Critical when <1.0 mg/L, Low when <2.0 mg/L
- **Sewage Indicators**: E.coli, enterococci, faecal indicators
- **BOD**: Biochemical Oxygen Demand
- **Ammonia**: Toxic to aquatic life
- **Suspended Solids**: Physical pollution
- **Oil/Grease**: Surface contamination

---

## 🛠️ Troubleshooting

### Common Issues

1. **"No EDM files could be parsed successfully"**
   - Ensure Excel files are in `xlsx/` folder
   - Check that files aren't corrupted
   - Verify you have all years (2020-2024)

2. **Missing dependencies error**
   - Run: `pip install pandas scipy numpy openpyxl`

3. **Memory errors with large CSV files**
   - The water quality CSVs can be 600MB+ each
   - Ensure you have at least 8GB RAM available
   - Close other applications while running

4. **No correlations found**
   - Verify both EDM and water quality data are for the same years
   - Check that coordinate columns are present in the data

### Data Quality Notes

- 2020 EDM data from individual water companies often lacks coordinates
- The scripts automatically skip files without valid location data
- Some overflow sites may not have nearby water quality samples

---

## 📈 Extending the Analysis

### Adjusting Parameters

In `batch_filter_final.py`:
```python
RADIUS_METERS = 5000  # Increase/decrease correlation radius
TARGET_POLLUTANTS_REGEX = r"..."  # Modify pollutants of interest
```

In `analyze_correlations_final.py`:
```python
TOP_N_RESULTS = 50  # Change number of results displayed
weights = {...}  # Adjust risk score component weights
```

### Adding Visualizations

The output CSVs are designed for easy visualization in:
- GIS software (QGIS, ArcGIS) using the easting/northing coordinates
- Python (matplotlib, plotly) for risk score distributions
- Mapping libraries (folium, keplergl) for interactive maps

---

## 📜 License

This project is released under the MIT License. The analysis uses publicly available data from the UK Environment Agency.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

For questions or issues, please open a GitHub issue.

---

## 🙏 Acknowledgments

- UK Environment Agency for providing open data
- Contributors to the pandas, scipy, and numpy projects

---

**Note:** This analysis is for research and accountability purposes. The identification of high-risk sites is based on correlation analysis and should be validated with field investigations before drawing definitive conclusions about causation.
