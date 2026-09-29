import pandas as pd

def load_data(path: str):
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format. Use CSV or XLSX.")
        print(f"[OK] Dataset loaded successfully -> {path}")
        print(f"[INFO] Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("[ERROR] File not found. Check the file path.")
        raise
    except ValueError as ve:
        print(f"[ERROR] {ve}")
        raise
    except Exception as e:
        print("[ERROR] Error while loading dataset:", e)
        raise