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


def audit_visual(df, name="Датасет"):
    """
    Красивый визуальный аудит датасета — понимается с одного взгляда.
    """
    print("=" * 60)
    print(f"📊 АУДИТ: {name}")
    print("=" * 60)

    # --- Размер ---
    rows, cols = df.shape
    print(f"\n📁 РАЗМЕР:      {rows} строк × {cols} столбцов")

    # --- Пропуски ---
    total_na = df.isna().sum().sum()
    total_cells = df.size
    pct = total_na / total_cells * 100
    print(f"📉 ПРОПУСКИ:    {total_na} ({pct:.1f}%)")

    # --- Дубликаты ---
    dups = df.duplicated().sum()
    print(f"🔁 ДУБЛИКАТЫ:   {dups}")

    # --- Выбросы (простые проверки) ---
    outliers = []
    if "age" in df.columns:
        age_num = pd.to_numeric(df["age"], errors="coerce")
        if (age_num < 0).any() or (age_num > 120).any():
            outliers.append("age")
    if "salary" in df.columns:
        sal_num = pd.to_numeric(df["salary"], errors="coerce")
        if (sal_num < 0).any() or (sal_num > 1_000_000).any():
            outliers.append("salary")
    if "experience" in df.columns:
        exp_num = pd.to_numeric(df["experience"], errors="coerce")
        if (exp_num < 0).any() or (exp_num > 60).any():
            outliers.append("experience")

    print(f"⚠️  ВЫБРОСЫ:     {', '.join(outliers) if outliers else 'нет'}")

    # --- Пропуски по столбцам (топ-8) ---
    print("\n" + "─" * 60)
    print("📋 ПРОПУСКИ ПО СТОЛБЦАМ (топ-8):\n")

    na_counts = df.isna().sum().sort_values(ascending=False).head(8)
    max_na = na_counts.max() if len(na_counts) > 0 else 1

    meaningful = ["termination_date", "last_promotion", "hire_date",
                  "employee_id", "full_name"]

    for col, cnt in na_counts.items():
        if cnt == 0:
            continue
        bar_len = int(cnt / max_na * 20)
        bar = "█" * bar_len
        if col in meaningful:
            mark = "⚠️  смысловой"
        else:
            mark = "✅ заполнить"
        print(f"  {col:<20} {bar:<20} {cnt:>4}  {mark}")

    # --- Типы данных ---
    print("\n" + "─" * 60)
    print("📋 ТИПЫ ДАННЫХ:\n")

    expected = {
        "age": "number", "experience": "number", "salary": "number",
        "projects_completed": "number", "satisfaction_score": "number",
        "hire_date": "datetime", "last_promotion": "datetime",
        "termination_date": "datetime", "is_manager": "bool"
    }

    correct, wrong = [], []
    for col, exp_type in expected.items():
        if col not in df.columns:
            continue
        actual = df[col].dtype
        ok = False
        if exp_type == "number" and pd.api.types.is_numeric_dtype(actual):
            ok = True
        elif exp_type == "datetime" and pd.api.types.is_datetime64_any_dtype(actual):
            ok = True
        elif exp_type == "bool" and actual == bool:
            ok = True

        if ok:
            correct.append(col)
        else:
            wrong.append(f"{col} ({actual} → {exp_type})")

    if correct:
        print(f"  ✅ Правильно:    {', '.join(correct)}")
    if wrong:
        print(f"  ⚠️  Исправить:   {', '.join(wrong)}")

    print("\n" + "=" * 60)