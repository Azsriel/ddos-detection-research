import pandas as pd
def reducedim(df1,df2):
	common_cols = list(set(df1.columns) & set(df2.columns))
	print("1")
	dfmixed_common = df1[common_cols]
	dfcic_common = df2[common_cols]
	newdf = pd.concat([dfmixed_common,dfcic_common],ignore_index=True)
	print(f"The number of common columns : {len(common_cols)}")
	return newdf

df1 = pd.read_parquet("CIC-DDoS2019-combined.parquet")
df2 = pd.read_csv("final_dataset.csv", nrows=100_000)
df3 = reducedim(df1,df2)
df3.to_parquet("final_shortened.parquet")
print("Converted")
