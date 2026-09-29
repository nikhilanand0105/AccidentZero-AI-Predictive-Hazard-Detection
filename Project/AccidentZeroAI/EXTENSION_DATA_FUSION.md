# Data Fusion + Image Extraction Extension (Additive)

This extension does **not** replace existing project logic.  
It adds a separate workflow that reads existing Excel output, extracts structured data from an image, and writes a **new** workbook.

## New module

- `utils/data_fusion_excel_extension.py`

## What it adds

1. Preserves original dataset by copying it unchanged into a new workbook (`Original Data`).
2. Extracts table/chart data from image:
   - Primary: Gemini vision structured OCR (if `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set).
   - Fallback: `pytesseract` OCR (if installed).
3. Maps extracted fields into meaningful columns (safety schema).
4. Creates separate datasets and sheets:
   - `Original Data`
   - `Image Extracted Data`
   - `Dataset_A`
   - `Dataset_B`
5. Performs per-dataset analysis:
   - Mean, Median, Mode, Std, Variance, Min, Max
   - Line, Bar, Histogram, Scatter charts
   - Graph interpretation notes
6. Performs data fusion (schema-aligned concatenation + dedup) into `Fused Dataset`.
7. Runs combined analysis on fused data:
   - Comparative line chart
   - Multi-variable bar chart
   - Correlation heatmap
8. Writes final cross-dataset comparison into `Final Comparison`.

## Run

```powershell
python utils/data_fusion_excel_extension.py `
  --input-excel data/safety_data.xlsx `
  --input-image data/your_input_image.png `
  --output-excel data/fusion_extension_output.xlsx
```

## Notes

- Existing files/sheets are untouched; output is written to a **new** workbook path.
- If chart libraries are missing, install project dependencies first:

```powershell
pip install -r requirements.txt
```
