def validate_data(df):
    report = {}
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicates"] = df.duplicated().sum()
    return report