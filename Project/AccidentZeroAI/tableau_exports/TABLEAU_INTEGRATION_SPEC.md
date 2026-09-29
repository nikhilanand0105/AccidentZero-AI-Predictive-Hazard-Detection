# Tableau Integration Spec - AccidentZero AI

## Objective
Integrate Tableau-based interactive visualization into the AccidentZero AI project to provide KPI, trend, and drill-down insights from CSV/Excel data and model outputs.

## Data Sources
- `tableau_exports/raw_safety_data.csv`
- `tableau_exports/batch_predictions.csv`
- `tableau_exports/model_metrics.csv`
- `tableau_exports/proposed_vs_existing.csv`
- `tableau_exports/risk_distribution.csv`
- `tableau_exports/risk_score_bands.csv`
- `tableau_exports/feature_summary_stats.csv`
- `tableau_exports/feature_correlation_long.csv`
- `tableau_exports/dataset_overview.csv`

## Scope Mapping
1. Data source integration
- Connect Tableau to the CSV exports listed above.
- Apply data type checks:
  - Numeric: risk scores, probabilities, counts.
  - Dimensions: model, risk level, feature names.
- Use relationships/joins only where needed:
  - `batch_predictions.csv` with `raw_safety_data.csv` by row index if comparative rows are needed.

2. Data modeling and preparation
- Calculated fields:
  - `Risk % = [ensemble_probability] * 100`
  - `High Risk Flag = IF [risk_score] >= 60 THEN 1 ELSE 0 END`
  - `Model Rank = RANK_DENSE([accuracy], 'desc')`
- Aggregations:
  - AVG risk score, COUNT rows, SUM high-risk flags.

3. Dashboard development
- Dashboard A: Dataset EDA
  - Class distribution, feature summary, correlation heatmap.
- Dashboard B: Model Performance
  - Accuracy/precision/recall/F1 by model.
- Dashboard C: Proposed vs Existing
  - Two-bar comparison with delta annotation.
- Dashboard D: Operational Risk
  - Risk level split, risk bands, top risky records.
- Add global filters:
  - Risk level, model, feature.

4. Visualization standards
- Use consistent colors:
  - LOW green, MODERATE amber, HIGH orange, CRITICAL red.
- Keep axis labels explicit (units where relevant).
- Show tooltips with key context fields.

5. Publishing and deployment
- Publish workbook to Tableau Public or Tableau Cloud.
- Copy share/embed URL from Tableau.

6. Project integration
- Frontend supports Tableau embed in `frontend/index.html` and `frontend/script.js`.
- Configure in `frontend/config.js`:
  - `TABLEAU_EMBED_URL`: your Tableau share URL.
  - `TABLEAU_AUTO_REFRESH_MS`: refresh interval in milliseconds.

7. Performance optimization
- Prefer Tableau Extracts for larger data.
- Keep calculations at source/tableau level minimal and reusable.
- Use filtered dashboards and reduce heavy sheet count per dashboard.

8. Testing and validation
- Cross-check Tableau totals against CSV source totals.
- Validate KPI values against backend outputs.
- Verify filter and drill-down behavior on desktop/mobile.

## Deliverables
- Tableau-ready datasets in `tableau_exports/`
- Embedded Tableau container in frontend
- Configurable refresh and embed settings
- Documentation for data, logic, and deployment
