import pandas as pd

df = pd.read_csv("data/processed_data/FinBERT_PyFin_Consistency.csv")

def get_conditions(row, model, sentiment_type, context_type):
    sentiment_col = f"{model}_{sentiment_type}_{context_type}"
    additive_col = f"{model}_additive_{context_type}"
    transitive_col = f"{model}_transitive_{context_type}"
    symmetric_col = f"{model}_symmetric_{context_type}"
    negation_col = f"{model}_negation_{context_type}"

    conditions = {
        "additive_match": row[sentiment_col] == row[additive_col],
        "transitive_match": row[sentiment_col] == row[transitive_col],
        "symmetric_match": row[sentiment_col] == row[symmetric_col],
        "negation_match": (row[sentiment_col] == "Neutral") & (row[negation_col] == "Neutral") & (row[sentiment_col] != row[negation_col]),
    }

    return conditions


df["PyFin_title_conditions"] = df.apply(lambda row: get_conditions(row, "PyFin", "sentiment", "title"), axis=1)
df["PyFin_content_conditions"] = df.apply(lambda row: get_conditions(row, "PyFin", "sentiment", "content"), axis=1)
df["FinBERT_title_conditions"] = df.apply(lambda row: get_conditions(row, "FinBERT", "sentiment", "title"), axis=1)
df["FinBERT_content_conditions"] = df.apply(lambda row: get_conditions(row, "FinBERT", "sentiment", "content"), axis=1)

models = ["PyFin", "FinBERT"]
content_types = ["title", "content"]
list_Columns = []

for model in models:
    for content_type in content_types:
        conditions_col = f"{model}_{content_type}_conditions"
        if conditions_col in df.columns:
            df[conditions_col] = df[conditions_col].apply(lambda x: {key: not val for key, val in x.items()})
        list_Columns.append(conditions_col)

flattened_columns = ["id", "original_content", "original_title"] + list_Columns

df = df[flattened_columns]

df.to_csv("data/processed_data/Conditions.csv", index=False)
