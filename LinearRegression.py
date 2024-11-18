import statsmodels.api as sm
import pandas as pd

df= pd.read_csv("data/processed_data/Enhanced_Disagreements.csv")

X = df[['title_length', 'content_length', 'title_number_count', 'content_number_count',
        'title_special_char_count', 'content_special_char_count', 'title_word_count',
        'content_word_count', 'title_avg_word_length', 'content_avg_word_length']]

X = sm.add_constant(X)

disagreement_columns = [col for col in df.columns if 'disagreement' in col]

for disagreement_col in disagreement_columns:
    Y = df[disagreement_col]

    model = sm.Logit(Y, X)
    result = model.fit()

    print(f"Regression Results for {disagreement_col}:")
    print(result.summary())
    print("\n" + "=" * 80 + "\n")
