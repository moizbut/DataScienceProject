import pandas as pd
df = pd.read_csv("urdu_ocr_dataset/labels.csv")
print(df['character'].value_counts())