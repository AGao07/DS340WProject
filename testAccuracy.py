import pandas as pd

df1 = pd.read_csv('data/processed_data/Classified_Finance_Post_Content_bygpt.csv')
df2 = pd.read_csv('data/processed_data/FinBERT_PyFin_Consistency.csv')

merged_df = pd.merge(df1, df2, on='id', how='inner')

def calculate_correctness(predictions, truth, category_col=None, is_negation=False):
    if is_negation:
        predictions = ["Bullish" if pred == "Bearish" else "Bearish" if pred == "Bullish" else pred for pred in predictions]
    total = len(predictions)

    neutral_correct = ((predictions == "Neutral") & (truth == "Neutral")).sum()
    bullish_correct = ((predictions == "Bullish") & (truth == "Bullish")).sum()
    bearish_correct = ((predictions == "Bearish") & (truth == "Bearish")).sum()

    neutral_correct_pct = neutral_correct / total * 100
    bullish_correct_pct = bullish_correct / total * 100
    bearish_correct_pct = bearish_correct / total * 100

    total_correct = (neutral_correct + bullish_correct + bearish_correct) / total * 100

    return neutral_correct_pct, bullish_correct_pct, bearish_correct_pct, total_correct



first_table_data = []

for model, pred_title_col, pred_content_col in [
    ("FinBERT", "FinBERT_sentiment_title", "FinBERT_sentiment_content"),
    ("PyFin", "PyFin_sentiment_title", "PyFin_sentiment_content")
]:
    for title in [True, False]:
        if title:
            predictions = merged_df[pred_title_col]
            truth = merged_df["category"]
        else:
            predictions = merged_df[pred_content_col]
            truth = merged_df["content_category"]

        total_correct = (predictions == truth).mean() * 100
        neutral_correct = ((predictions == "Neutral") & (truth == "Neutral")).mean() * 100
        bullish_correct = ((predictions == "Bullish") & (truth == "Bullish")).mean() * 100
        bearish_correct = ((predictions == "Bearish") & (truth == "Bearish")).mean() * 100

        first_table_data.append({
            "Model": model,
            "Title": title,
            "Neutral Correct (%)": neutral_correct,
            "Bullish Correct (%)": bullish_correct,
            "Bearish Correct (%)": bearish_correct,
            "Total Correct (%)": total_correct,
        })

    first_table = pd.DataFrame(first_table_data)
    print(first_table)
second_table_data = []

for model_prefix, neg_col, sym_col, add_col, trans_col, pred_title_col, pred_content_col in [
    ("FinBERT", "FinBERT_negation_title", "FinBERT_symmetric_title", "FinBERT_additive_title", "FinBERT_transitive_title",
     "FinBERT_sentiment_title", "FinBERT_sentiment_content"),
    ("PyFin", "PyFin_negation_title", "PyFin_symmetric_title", "PyFin_additive_title", "PyFin_transitive_title",
     "PyFin_sentiment_title", "PyFin_sentiment_content")
]:
    for category, col_name in [
        ("negation", neg_col), ("symmetric", sym_col), ("additive", add_col), ("transitive", trans_col)
    ]:
        for title in [True, False]:
            if title:
                predictions = merged_df[pred_title_col]
                truth = merged_df["category"]
            else:
                predictions = merged_df[pred_content_col]
                truth = merged_df["content_category"]

            neutral_correct, bullish_correct, bearish_correct, total_correct = calculate_correctness(
                predictions,
                truth,
                is_negation=(category == "negation")
            )

            second_table_data.append({
                "Category": f"{model_prefix}_{category}",
                "Title": title,
                "Neutral Correct (%)": neutral_correct,
                "Bullish Correct (%)": bullish_correct,
                "Bearish Correct (%)": bearish_correct,
                "Total Correct (%)": total_correct,
            })

second_table = pd.DataFrame(second_table_data)
first_table.to_csv('data/processed_data/baseline_accuracy_results.csv', index=False)
second_table.to_csv('data/processed_data/consistency_accuracy_results.csv', index=False)
