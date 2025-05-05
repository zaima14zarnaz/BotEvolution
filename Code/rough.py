import pandas as pd
from dataset import DatasetHandler

# # Load the CSV file
# # df = pd.read_csv('/home/zaimaz/Desktop/research1/BotEvolution/Dataset/twibot22/tweet_2-008_humans.csv')
# fname = '/home/zaimaz/Desktop/research1/BotEvolution/Dataset/twibot22/tweet_2-008_humans.csv'
# dataset = DatasetHandler(addr=fname)
# dataset.view_dataset()
# df = dataset.df
# # df = df.drop('Humor Label', axis=1)


# # df.to_csv('/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci_data_analysis_human.csv', index=False)

# # Convert 'created_at' to datetime format
# # df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
# df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')


# # Extract the year
# df['year'] = df['created_at'].dt.year

# # Count rows per year
# year_counts = df['year'].value_counts().sort_index()

# # Print the counts
# print(year_counts)

# llm_folder_path_1 = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/LLM_tweets/7 Batch 3001-4700"
# combined_file_path = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/LLM_tweets/batch_3001_4700.csv"
# DatasetHandler.combine_csv_files(llm_folder_path_1, combined_file_path)


# dataset = DatasetHandler(combined_file_path)
# dataset.clean_dataset(tweet_col_name='humanized_tweet')
# dataset.view_dataset()
# DatasetHandler.write_to_csv(df=dataset.df, csv_addr=combined_file_path, header=True)

f1 = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/llm_tweets_analysis.csv"
f2 = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/llm_tweets_analysis_merged.csv"
f3 = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/llm_tweets_analysis_set3.csv"

cols = ['Created At', 'Grammatical Accuracy','Humor']

df1 = pd.read_csv(f2, usecols=cols)
df2 = pd.read_csv(f3, nrows=100000, usecols=cols)
# print(df1.columns)
# df2 = pd.read_csv(f2)
df = pd.concat([df1,df2], ignore_index=True)

df.to_csv(f3, index=False, mode='w')
df = pd.read_csv(f3)
print(len(df))
print(df.head)