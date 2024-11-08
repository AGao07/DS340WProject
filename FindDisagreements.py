import pandas as pd

df = pd.read_csv("data/processed_data/Conditions.csv")

models = ["PyFin", "FinBERT"]
content_types = ["title", "content"]

disagreement_columns = []

for model in models:
    for content_type in content_types:
        pyfin_condition_col = f"{model}_title_conditions" if content_type == "title" else f"{model}_content_conditions"
        finbert_condition_col = f"FinBERT_{content_type}_conditions"
        disagreement_col = f"{model}_vs_FinBERT_{content_type}_disagreement"
        df[disagreement_col] = df.apply(lambda row: row[pyfin_condition_col] != row[finbert_condition_col], axis=1)
        disagreement_columns.append(disagreement_col)

disagreements = df[df[disagreement_columns].any(axis=1)]
disagreements.to_csv("data/processed_data/Disagreements.csv", index=False)
