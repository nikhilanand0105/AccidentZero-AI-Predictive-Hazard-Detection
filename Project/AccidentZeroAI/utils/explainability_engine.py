def generate_explanation(row):
    reasons = []
    if row["fatigue_index"] > 1:
        reasons.append("Elevated worker fatigue detected")
    if row["equipment_risk"] > 0:
        reasons.append("Equipment condition indicates potential risk")
    if row["weather_severity"] > 0:
        reasons.append("Environmental conditions contributing to risk")
    if row.get("safety_compliance_risk", 0) < 0:
        reasons.append("Safety compliance weaker relative to workload")
    if not reasons:
        reasons.append("No significant risk factors detected")
    return reasons