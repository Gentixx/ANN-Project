import pandas as pd

df=pd.read_csv("data.tsv",sep="\t",
               usecols=["product_name","brands","additives"])

df=df.dropna()