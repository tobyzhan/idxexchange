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

### Files
```
data/
├── CRMLSSold202412.csv            # December 2024
├── CRMLSSold202501_filled.csv     # January 2025
├── CRMLSSold202502.csv            # February 2025
├── CRMLSSold202503.csv            # March 2025
├── CRMLSSold202504.csv            # April 2025
└── CRMLSSold202505.csv            # May 2025
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
- **Numeric imputation**: Filled missing values with **median** (robust to right-skewed price distribution)
- **Categorical imputation**: Filled missing values with **mode** (preserves existing distribution)
- **Outlier removal**: Applied IQR method (1.5×IQR) to 6 key features:
  - `ClosePrice`, `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, `LotSizeAcres`, `YearBuilt`

### Step 3 – Feature Engineering
- One-hot encoded `City`, `CountyOrParish`, `PostalCode`
- Applied `SimpleImputer` post-encoding to handle sparse columns (`FireplacesTotal` dropped — all NaN)
- Applied `StandardScaler` for neural network training

---

## Models & Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **Ridge Regression** | **0.8619** | $187,503.89 | $129,317.99 |
| Random Forest | 0.8014 | $224,846.93 | $151,872.29 |
| XGBoost | 0.8364 | $204,098.07 | $138,985.64 |
| Neural Network | 0.8798 | $174,925.59 | $109,659.47 |

> ✅ **Best Performer: Neural Network** with R² = 0.8798 and 88.46% accuracy (MAPE = 11.54%)

### Neural Network Architecture
```
Input (~2143 features)
  → Linear(512) → ReLU → BatchNorm → Dropout(0.3)
  → Linear(256) → ReLU → BatchNorm → Dropout(0.3)
  → Linear(128) → ReLU → BatchNorm → Dropout(0.3)
  → Linear(64)  → ReLU → BatchNorm → Dropout(0.3)
  → Linear(1)   [Price Output]
```
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau
- **Loss**: MSELoss
- **Epochs**: 50 | **Batch Size**: 256

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

---

## Key Findings

### Top 10 Most Important Features (Neural Network)
| Rank | Feature | Importance |
|---|---|---|
| 1 | LivingArea | 0.2421 |
| 2 | LotSizeAcres | 0.0864 |
| 3 | BathroomsTotalInteger | 0.0672 |
| 4 | GarageSpaces | 0.0632 |
| 5 | CountyOrParish_Riverside | 0.0553 |
| 6 | DaysOnMarket | 0.0510 |
| 7 | YearBuilt | 0.0498 |
| 8 | CountyOrParish_San Bernardino | 0.0406 |
| 9 | CountyOrParish_Santa Clara | 0.0295 |
| 10 | City_Lancaster | 0.0294 |

### Insights
- **Neural Network outperformed all other models** — R² of 0.8798 vs Ridge's 0.8619
- **LivingArea dominates** as the single most important predictor (importance = 0.242)
- **Location features confirm** the real estate principle of "location, location, location" — County features appear in top 10
- **Ridge outperformed both Random Forest and XGBoost** — suggesting the one-hot encoded feature space benefits from linear regularization
- **Random Forest was the weakest performer** at R² = 0.8014, likely due to limited tree depth (max_depth=10)

---

## Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch
```

## Usage
Open and run `CA_house_prediction.ipynb` top to bottom. Ensure all 6 CSV data files are present in the `data/` directory.

---

## Future Improvements
- Hyperparameter tuning with Optuna or GridSearchCV on all models
- Target encoding for high-cardinality categoricals (City, PostalCode)
- Additional features: school ratings, walkability scores, crime index
- Cross-validation instead of single train/test split
