import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
data = {
    "shift_hours": np.random.randint(6, 13, n),
    "overtime_hours": np.random.randint(0, 5, n),
    "worker_experience": np.random.randint(1, 15, n),
    "equipment_age": np.random.randint(1, 10, n),
    "maintenance_score": np.random.randint(50, 100, n),
    "temperature": np.random.randint(22, 40, n),
    "humidity": np.random.randint(35, 85, n),
    "inspection_score": np.random.randint(55, 100, n),
    "accident":np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
}
df = pd.DataFrame(data)
df.to_excel("data/safety_data.xlsx", index=False)
print("[OK] Excel file filled with 1000 realistic rows")