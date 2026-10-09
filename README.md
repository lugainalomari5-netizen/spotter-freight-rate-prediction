# Spotter Freight Rate Prediction

## Project Overview

This project predicts freight load posted rates using historical shipment data. It includes exploratory data analysis, data cleaning, time-based model evaluation, final validation predictions, and a December 2025 pricing chart for a fixed shipment.

## Dataset

- **Training dataset:** 48,000 labeled freight loads from January–October 2025.
- **Validation dataset:** 12,000 unlabeled loads from November–December 2025.
- **December chart:** 31 daily predictions for a fixed Lexington → Fort Wayne shipment.

The prediction target is `posted_rate`.

## Approach

### Exploratory Data Analysis

The analysis examined freight rates, distances, equipment types, missing values, invalid weights, and temporal patterns.

Notable findings included:

- Freight distance has a strong positive correlation with posted rate.
- Some records contain missing or nonpositive weights.
- Equipment types have different average rate-per-mile values.

### Data Preprocessing

- Converted invalid, nonpositive weights to missing values.
- Imputed missing weights and market indices using training-set medians.
- Extracted month, day of week, and day of month from shipment dates.
- One-hot encoded equipment categories.
- Used geographic coordinates and shipment characteristics as predictive features.

### Model

The main model is a `RandomForestRegressor` with:

- `n_estimators=100`
- `max_depth=15`
- `min_samples_leaf=3`
- `random_state=42`

The model was evaluated using a chronological split: January–September 2025 for training and October 2025 for testing. This avoids training on later observations when evaluating earlier predictions.

## October 2025 Evaluation

| Model | MAE | RMSE |
|---|---:|---:|
| Training-mean baseline | $1,179.80 | $1,528.57 |
| Random Forest | **$150.72** | **$661.73** |

The Random Forest reduced MAE by approximately **87.23%** compared with the training-mean baseline.

### Limitations

The model substantially underpredicts some unusually expensive loads. Its October MAE for loads above the training-set 95th-percentile price threshold was $899.17, with an average prediction bias of approximately −$732.26.

Performance on the unlabeled final validation dataset is unknown and will be evaluated by Spotter.

## Final Validation Predictions

The main model was retrained on all 48,000 labeled observations to generate 12,000 predictions.

Output: `validation_predictions.csv`

Required columns:

- `load_id`
- `predicted_rate`

## December Pricing Chart

The December chart input contains 31 dates for a fixed shipment:

- Pickup: Lexington
- Delivery: Fort Wayne
- Distance: 360 miles
- Equipment: Dry Van
- Weight: 32,000

Because this input lacks geographic coordinates, `market_index`, and `quote_signal`, a separate Random Forest model was trained using only the available shipment and date features.

Its October evaluation results were:

- MAE: **$156.85**
- RMSE: **$670.16**

The completed December predictions are saved in `data/december_chart_inputs.csv`, with a backup in `outputs/december_predictions.csv`.

The generated chart is saved at `scorer_results/candidate_december.png`.

December predictions should be interpreted cautiously because they extend beyond the model's historical training period.

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Evaluate the main model:

```bash
python src/train_predict.py
```

Generate final validation predictions:

```bash
python src/predict_validation.py
```

Generate December predictions:

```bash
python src/predict_december.py
```

Validate the submission and generate the December chart:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Submission Validation

The provided scorer successfully validated:

- 12,000 final validation predictions.
- 31 fixed December predictions.
- Generation of `scorer_results/candidate_december.png`.

The scorer checks submission validity; final validation accuracy is calculated separately by Spotter.
