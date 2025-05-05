from dataset import DatasetHandler
import pandas as pd
import numpy as np

skip_rows = 0
load_rows = 420000

encoding = None # 'ISO-8859-1'
dataset_addr = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/LLM_tweets/batch_3001_4700.csv"
main_dataset = DatasetHandler(dataset_addr, encoding=encoding)
main_dataset.view_dataset()
main_dataset.clean_dataset(tweet_col_name='humanized_tweet')

cleaned_addr = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/LLM_tweets/cleaned_batch_3001_4700.csv"
DatasetHandler.write_to_csv(df=main_dataset.df, csv_addr=cleaned_addr)
# # Convert 'created_at' to datetime
# main_dataset.df['created_at'] = pd.to_datetime(main_dataset.df['created_at'], errors='coerce')

# # Extract year
# main_dataset.df['year'] = main_dataset.df['created_at'].dt.year

# # Set xi = number of rows per year
# xi_dict = {
#     2007:      76,
#     2008:     500,
#     2009:    9000,
#     2010:    13100,
#     2011:    27200,
#     2012:    28570,
#     2013:    32570,
#     2014:    32570,
#     2015:    32570,
#     2016:    32570,
#     2017:    32570,
#     2018:    32570,
#     2019:    32570,
#     2020:    32570,
#     2021:    32570,
#     2022:    34570
# }

# # Sample from each year using the custom counts
# sampled_dfs = []
# for year, xi in xi_dict.items():
#     df_year = main_dataset.df[main_dataset.df['year'] == year]
#     if len(df_year) >= xi:
#         sampled = df_year.sample(n=xi, random_state=42)
#         sampled_dfs.append(sampled)
#     else:
#         print(f"⚠️  Not enough rows for {year}. Available: {len(df_year)} < Requested: {xi}")

# # Combine and shuffle
# final_df = pd.concat(sampled_dfs, ignore_index=True)
# final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # final_df = main_dataset.df[main_dataset.df['Bot Label'] == 0]

# Save to CSV
# final_df.to_csv(cleaned_addr, index=False)

# print("✅ Final sampled DataFrame shape:", final_df.shape)
cleaned_dataset = DatasetHandler(cleaned_addr, encoding='ISO-8859-1')



   
analysis_csv =   "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/llm_tweets_analysis_set3.csv"
df2 = pd.DataFrame({
    # 'User ID': main_dataset.df['user_id'],
    # 'Tweet ID': main_dataset.df['id'],
    # 'Created At': main_dataset.df['created_at'],
    
    # 'User ID': cleaned_dataset.df['author_id'],
    # 'Tweet ID': cleaned_dataset.df['id'],
    'Created At': cleaned_dataset.df['created_at'],
    'Grammatical Accuracy': np.nan,  # or '' if you prefer
    'Humor':np.nan
})
# df2.drop('Created At', axis=1)
DatasetHandler.write_to_csv(df2, analysis_csv, mode='w', header=True)  
analysis_dataset = DatasetHandler(analysis_csv, encoding='ISO-8859-1')
analysis_dataset.view_dataset()


# dataset_addr = '/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci_data_analysis_bots.csv'
# df = pd.read_csv(dataset_addr, encoding='ISO-8859-1')
# print(df.columns)

# # Drop the unwanted column (e.g., 'Grammatical Accuracy')
# df = df.drop('Tweet', axis=1)

# # Save it back to CSV (overwrite or create a new one)
# df.to_csv(dataset_addr, index=False)