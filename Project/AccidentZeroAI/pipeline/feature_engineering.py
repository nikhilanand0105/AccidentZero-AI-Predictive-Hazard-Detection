def engineer_features(df):
    df["fatigue_index"] = df["shift_hours"] + df["overtime_hours"]
    df["equipment_risk"] = df["equipment_age"] / (df["inspection_score"] + 1)
    df["weather_severity"] = df["temperature"] * df["humidity"]
    return df