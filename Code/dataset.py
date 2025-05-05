import pandas as pd
import re
import os
import glob
from langdetect import detect, LangDetectException

class DatasetHandler:
    def __init__(self, addr, nrows=None, skiprows=None, encoding=None):
        if skiprows is None:
            skiprows = -1
        if encoding is None:
            if nrows is None:
                self.df = pd.read_csv(addr, skiprows=range(1, skiprows+1))
            else:
                self.df = pd.read_csv(addr, nrows=nrows, skiprows=range(1, skiprows+1))
        else:
            if nrows is None:
                self.df = pd.read_csv(addr, skiprows=range(1, skiprows+1), encoding=encoding)
            else:
                self.df = pd.read_csv(addr, nrows=nrows, skiprows=range(1, skiprows+1), encoding=encoding)
    
    def view_dataset(self):
        print(f'Columns: {self.df.columns}')
        print(self.df.head())
        print(f"Size of dataset: {len(self.df)}")
        # print(self.df['source'])
    
    def get_column_nammes(self):
        return self.df.columns
        
    
    @staticmethod
    def write_to_csv(df, csv_addr, mode=None, header=False):
        if mode is None:
            df.to_csv(csv_addr, index=False, header=True)
        else:
            df.to_csv(csv_addr, mode=mode, index=False, header=header)
    
    def is_english(self, text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False  # in case the text is empty or invalid

    def clean_text(self, text):
        text = re.sub(r"http\S+", "", text)       # Remove URLs
        text = re.sub(r"@\w+", "", text)          # Remove @mentions
        text = re.sub(r"#\w+", "", text)          # Remove hashtags
        text = re.sub(r"[^A-Za-z0-9\s.,!?]", "", text)  # Remove special characters
        text = text.strip()
        return text
    
    def clean_dataset(self, tweet_col_name):
        self.df[tweet_col_name] = self.df[tweet_col_name].astype(str).apply(self.clean_text)
        self.df = self.df[self.df[tweet_col_name].astype(str).str.strip() != '']
        # # Apply language detection and filter
        # self.df['is_english'] = self.df['text'].astype(str).apply(self.is_english)

        # # Keep only English tweets
        # self.df = self.df[self.df['is_english']].drop(columns=['is_english']).reset_index(drop=True)
    
    @staticmethod
    def combine_csv_files(folder_path, combined_file):
        # Use glob to get all CSV file paths in the folder
        csv_files = sorted(glob.glob(os.path.join(folder_path, "batch_*.csv")))

        # Read and concatenate all CSVs
        combined_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        # Save to a single CSV file with header
        combined_df.to_csv(combined_file, index=False, header=True)



# dataset_addr = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/tweets.csv"
# dataset = DatasetHandler(dataset_addr, nrows=2000000, encoding='ISO-8859-1')
# dataset.view_dataset()
# dataset.clean_dataset()
# # dataset.df['created_at'] = pd.to_datetime(dataset.df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')

# # Filter rows where the year is 2014
# # df_2014 = dataset.df[dataset.df['created_at'].dt.year == 2014]

# # Print matching rows
# # print(df_2014)
# dataset.write_to_csv(dataset.df, csv_addr="/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci-2017.csv/cleaned_social_spambots_2.csv")


# analysis_csv =   "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci_data_analysis_bots.csv"
# df2 = pd.DataFrame(columns=['User ID', 'Tweet ID', 'Tweet', 'Created At', 'Grammatical Accuracy', 'Sentence Complexity'])
# df2['User ID'] = dataset.df['user_id']
# df2['Tweet ID'] = dataset.df['id']
# # df2['Tweet'] = dataset.df['text']
# df2['Created At'] = dataset.df['created_at']
# DatasetHandler.write_to_csv(df2, analysis_csv)  


# dataset_addr = '/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci_data_analysis_bots.csv'
# df = pd.read_csv(dataset_addr, encoding='ISO-8859-1')
# print(df.columns)

# # Drop the unwanted column (e.g., 'Grammatical Accuracy')
# df = df.drop('Tweet', axis=1)

# # Save it back to CSV (overwrite or create a new one)
# df.to_csv(dataset_addr, index=False)

# column_names = ['User ID', 'Tweet ID', 'Created At', 'Grammatical Accuracy', 'Sentence Complexity']

# # Load only the first 20,000 rows
# df = pd.read_csv(analysis_csv, nrows=20000, encoding='ISO-8859-1')

# # Save back to the same file (overwrite)
# df.to_csv(analysis_csv, index=False, header=True)