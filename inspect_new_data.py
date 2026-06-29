import pandas as pd
import unicodedata
import sys

# force utf-8 output to avoid charmap errors
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel(r'P:\MVR\new_data_exp\XRF_Analysis_Report.xlsx')

cols = [unicodedata.normalize('NFKC', c) for c in df.columns]
df.columns = cols

print("Columns:", cols)
print("Data Types:\n", df.dtypes)

for i, row in df.head(3).iterrows():
    print(row.to_dict())
