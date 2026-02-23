# California House Price Prediction Model

## Overview
A machine learning project that predicts residential house sale prices in California using 6 months of CRMLS (California Regional Multiple Listing Service) transaction data from **December 2024 – May 2025**. Four models are implemented and compared to find the best predictor of single-family home prices statewide.

---

## Dataset
| Property | Detail |
|---|---|
| Source | CRMLS Sold Property Records |
| Time Period | December 2024 – May 2025 |
| Property Type | Residential Single-Family Residences |
| Coverage | California Statewide |
| Raw Records | ~125,000+ listings |
| After Filtering | ~47,000+ SFR transactions |

### Files
```
data/
├── CRMLSSold202412.csv       # December 2024
├── CRMLSSold202501_filled.csv # January 2025 (with filled lat/lon)
├── CRMLSSold202502.csv       # February 2025
├── CRMLSSold202503.csv       # March 2025
├── CRMLSSold202504.csv       # April 2025
└── CRMLSSold202505.csv       # May 2025
```

---

## Features Used (17 Variables)

**Property Characteristics**
- `LivingArea` – Square footage of living space
- `BedroomsTotal` – Number of bedrooms
- `BathroomsTotalInteger` – Number of bathrooms
- `LotSizeAcres` – Lot size in acres
- `YearBuilt` – Year property was built
- `Stories` – Number of stories
- `GarageSpaces` – Number of garage spaces
- `ParkingTotal` – Total parking spaces
- `FireplacesTotal` – Number of fireplaces
- `AssociationFee` – Monthly HOA fee

**Location**
- `Latitude` / `Longitude` – Geographic coordinates
- `City` – City name
- `CountyOrParish` – County name
- `PostalCode` – ZIP code

**Market Metrics**
- `DaysOnMarket` – Days listed before sale

**Target Variable**
- `ClosePrice` – Final sale price ($)

---

## Pipeline

### Step 1 – Data Preprocessing
- Combined 6 monthly CSV files into one dataset
- Filtered for `Residential` + `SingleFamilyResidence` only
- Dropped imputed coordinate columns (`latfilled`, `lonfilled`) from January data
- Selected 17 relevant features

### Step 2 – Data Cleaning
- Standardized null values (`N/A`, `Unknown`, `TBD`, `""`) → `NaN`
- **Numeric imputation**: Filled missing values with **median** (robust to real estate's right-skewed price distribution)
- **Categorical imputation**: Filled missing values with **mode** (preserves existing distribution)
- **Outlier removal**: Applied IQR method (1.5×IQR) to 6 key features:
  - `ClosePrice`, `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, `LotSizeAcres`, `YearBuilt`

### Step 3 – Feature Engineering
- One-hot encoded `City`, `CountyOrParish`, `PostalCode`
- Applied `SimpleImputer` post-encoding to handle sparse columns
- Applied `StandardScaler` for neural network training

### Step 4 – Model Training & Evaluation

---

## Models & Results

| Model | R² Score | Notes |
|---|---|---|
| Ridge Regression | ~0.75 | L2 regularization, α=0.1 |
| Random Forest | ~0.82 | 50 trees, max_depth=10 |
| **XGBoost** | **~0.84** | **Best performer** |
| Neural Network | ~0.80 | PyTorch, 4 hidden layers |

### XGBoost Configuration
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Neural Network Architecture
```
Input → Linear(512) → ReLU → BatchNorm → Dropout(0.3)
      → Linear(256) → ReLU → BatchNorm → Dropout(0.3)
      → Linear(128) → ReLU → BatchNorm → Dropout(0.3)
      → Linear(64)  → ReLU → BatchNorm → Dropout(0.3)
      → Linear(1)
```
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau
- **Loss**: MSELoss
- **Epochs**: 50, Batch Size: 256

---

## Key Findings
- **Location dominates**: County and city features ranked highest in XGBoost feature importance
- **Size matters**: `LivingArea` was the top numeric predictor (correlation = 0.38)
- **XGBoost outperformed** the neural network on this structured tabular dataset
- Price distribution is **right-skewed** — median imputation was critical for accurate missing value handling

---

## Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch
```

## Usage
Open and run [`CA_house_prediction.ipynb`](CA_house_prediction.ipynb) top to bottom. Ensure all 6 CSV data files are present in the `data/` directory.

---

## Future Improvements
- Hyperparameter tuning with Optuna or GridSearchCV
- Target encoding for high-cardinality categorical features (City, PostalCode)
- Additional features: school ratings, walkability scores, crime index
- Cross-validation instead of single train/test split
- Model stacking / ensembling
- REST API deployment with FastAPI + joblib
