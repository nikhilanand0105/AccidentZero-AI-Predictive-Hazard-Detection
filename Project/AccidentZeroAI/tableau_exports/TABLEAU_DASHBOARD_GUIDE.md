# Tableau Dashboard Build Guide (AccidentZero AI)

Use these files in Tableau:
- raw_safety_data.csv
- batch_predictions.csv
- model_metrics.csv
- proposed_vs_existing.csv
- risk_distribution.csv
- risk_score_bands.csv
- feature_summary_stats.csv
- feature_correlation_long.csv
- dataset_overview.csv

Recommended dashboards:
1) Dataset EDA Dashboard
- Sheets: Accident class split, numeric distribution, feature summary table

2) Model Performance Dashboard
- Sheets: Accuracy/Precision/Recall/F1 by model

3) Proposed vs Existing Dashboard
- Sheets: 2-bar comparison chart + delta text

4) Risk Intelligence Dashboard
- Sheets: Risk level distribution, risk score bands, top risky rows from batch_predictions

5) Correlation Heatmap Dashboard
- Use feature_correlation_long.csv (feature_x, feature_y, correlation)

Tip: Format percentages to 2 decimals and use consistent colors:
- LOW: green, MODERATE: amber, HIGH: orange, CRITICAL: red.
