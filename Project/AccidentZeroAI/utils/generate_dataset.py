import pandas as pd
import numpy as np

def generate_dataset(n=500):
    np.random.seed(42)
    data = {
        "shift_hours": np.random.randint(6, 13, n),
        "overtime_hours": np.random.randint(0, 5, n),
        "worker_experience": np.random.randint(1, 15, n),
        "equipment_age": np.random.randint(1, 10, n),
        "maintenance_score": np.random.randint(40, 100, n),
        "temperature": np.random.randint(20, 45, n),
        "humidity": np.random.randint(30, 90, n),
        "inspection_score": np.random.randint(50, 100, n)
    }
    df = pd.DataFrame(data)
    df["accident"] = np.random.choice(
        [0, 1],
        size=len(df),
        p=[0.6, 0.4]
    )
    return df

df = generate_dataset()
df.to_csv("data/safety_data.csv", index=False)
print("[OK] Dataset generated successfully")