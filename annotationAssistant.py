import pandas as pd
import os

#by hand
def annotate_csv(input_csv, output_csv):
    input_df = pd.read_csv(input_csv)
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=["Id", "annotationtitle", "annotationcontent"]).to_csv(output_csv, index=False)
    output_df = pd.read_csv(output_csv)
    annotated_ids = set(output_df["Id"])
    rows_to_annotate = input_df[~input_df["id"].isin(annotated_ids)]

    for _, row in rows_to_annotate.iterrows():
        print(f"Title: {row['title']}")
        title_annotation = input("Enter annotation for the title (bullish, neutral, bearish): ").strip().lower()

        print(f"Content: {row['content']}")
        content_annotation = input("Enter annotation for the content (bullish, neutral, bearish): ").strip().lower()

        new_row = {
            "Id": row["id"],
            "annotationtitle": title_annotation,
            "annotationcontent": content_annotation
        }
        output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
        output_df.to_csv(output_csv, index=False)
        print(f"Annotations for ID {row['id']} saved.\n")

annotate_csv('data/processed_data/reddit_finance_posts_cleaned.csv', 'data/processed_data/Annotations.csv')
