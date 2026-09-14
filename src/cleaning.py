import pandas as pd
import numpy as np


def load_data(path, sheet_name):
    return pd.read_excel(path, sheet_name=sheet_name)


def audit(df, name="Датасет"):
    print(f"\n=== АУДИТ: {name} ===")
    print(f"Размер: {df.shape}")
    print(df.dtypes)
    print(df.isna().sum())
    print(f"Дубликатов: {df.duplicated().sum()}")
    print(df.describe())


def fix_types(df):
    df = df.copy()
    df["salary"] = (df["salary"].astype(str)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False))
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    mapping = {"True": True, "False": False, "1": True, "0": False,
               "Yes": True, "No": False, "yes": True, "no": False}
    df["is_manager"] = df["is_manager"].map(lambda x: mapping.get(str(x), np.nan))
    for col in ["age", "experience", "projects_completed", "satisfaction_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["hire_date", "last_promotion", "termination_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_categories(df):
    df = df.copy()
    df["gender"] = (df["gender"].astype(str).str.strip().str.lower()
                    .replace({"m": "Male", "f": "Female", "male": "Male",
                              "female": "Female", "nan": np.nan}))
    df["city"] = df["city"].astype(str).str.strip().replace({
        "СПб": "Санкт-Петербург", "Питер": "Санкт-Петербург",
        "Екб": "Екатеринбург", "Новосиб": "Новосибирск",
        "Kazan": "Казань", "Moscow": "Москва",
        "Нижний": "Нижний Новгород", "nan": np.nan})
    for col in ["department", "position"]:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    return df


def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"Удалено дубликатов: {before - len(df)}")
    return df.reset_index(drop=True)


def fill_missing(df):
    df = df.copy()
    for col in ["age", "experience", "salary", "projects_completed", "satisfaction_score"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["gender", "city", "department", "position"]:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
    df["is_manager"] = df["is_manager"].fillna(False).astype(bool)
    return df


def remove_outliers(df):
    df = df.copy()
    df = df[(df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 120))]
    for col in ["experience", "salary", "projects_completed", "satisfaction_score"]:
        df = df[(df[col].isna()) | (df[col] >= 0)]
    q1, q3 = df["salary"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["salary"] = df["salary"].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df.reset_index(drop=True)


def clean_pipeline(df):
    df = fix_types(df)
    df = normalize_categories(df)
    df = drop_duplicates(df)
    df = fill_missing(df)
    df = remove_outliers(df)
    return df.reset_index(drop=True)