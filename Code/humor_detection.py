from transformers import pipeline
import time
from dataset import DatasetHandler
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np


class HumorDetector:
    def __init__(self, main_df, output_df):
        # Initialize the text classification pipeline with the humor detection model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classifier = pipeline("text-classification", model="mohameddhiab/humor-no-humor", device=self.device)
        self.analysis_csv_df = output_df
        self.main_df = main_df
    
    def humor_score(self, text):
        # Perform classification
        return self.classifier(text)
    
    def analyze_humor(self, csv_addr, tweet_col_name, start_from):
        batch_size = 10000
        total_texts = len(self.main_df) - start_from
        texts = list(self.main_df[tweet_col_name][start_from:])
        texts = [t if isinstance(t, str) else '' for t in texts]

        humor_labels = []

        for batch_start in range(0, total_texts, batch_size):
            batch_end = batch_start + batch_size
            batch_texts = texts[batch_start:batch_end]

            print(f"Processing batch {batch_start + start_from} to {min(batch_end + start_from, len(self.main_df))}...", flush=True)
            start_time = time.time()
            results = self.classifier(batch_texts)
            end_time = time.time()
            print(f"Batch processed in {round(end_time - start_time, 2)} seconds", flush=True)
            humor_labels.extend(['1' if result['label'] == 'HUMOR' else '0' for result in results])
            
            self.analysis_csv_df.loc[start_from:start_from + len(humor_labels) - 1, 'Humor'] = humor_labels

            # Write results to CSV
            DatasetHandler.write_to_csv(self.analysis_csv_df, csv_addr)

    @staticmethod
    def plot_humor_1(csv_bot1,
                csv_bot2, 
                csv_human1,
                csv_human2,
                plot_save_path,
                user_types, 
                plot_period='M', 
                target_col_name='Humor',
                plot_title='Bot vs Human Humor Over Time',
                x_label='Time (Monthly)',
                y_label='Average Accuracy Score'):

        # Load CSVs
        dfs = [pd.read_csv(path) for path in [csv_bot1, csv_bot2, csv_human1, csv_human2]]

        # Parse datetime and remove timezone
        for df in dfs:
            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce').dt.tz_localize(None)

        # Concatenate datasets
        df_bot = pd.concat([dfs[0],dfs[1]],ignore_index=True)[['Created At', target_col_name]].dropna()
        df_human = pd.concat([dfs[2],dfs[3]],ignore_index=True)[['Created At', target_col_name]].dropna()

        print(df_bot.head)
        print(df_human.head)

        df_bot = df_bot[df_bot['Created At'].dt.year > 2008]
        df_human = df_human[df_human['Created At'].dt.year > 2008]

        # Label type
        df_bot['Type'] = user_types[0]
        df_human['Type'] = user_types[1]

        # Combine all
        df_all = pd.concat([df_bot, df_human], ignore_index=True)

        # Define period
        period_format = '%Y-%m' if plot_period == 'M' else '%Y'
        df_all['Period'] = df_all['Created At'].dt.to_period(plot_period).dt.to_timestamp()

        # Calculate rolling average
        window_size = 6
        df_all[target_col_name] = df_all.groupby('Type')[target_col_name].transform(
            lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean()
        )

        # Plot
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_all, x='Period', y=target_col_name, hue='Type', errorbar=('ci', 95), marker='.')

        # Add linear trend lines
        for user_type in df_all['Type'].unique():
            df_subset = df_all[df_all['Type'] == user_type]
            z = np.polyfit(mdates.date2num(df_subset['Period']), df_subset[target_col_name], 1)
            p = np.poly1d(z)
            plt.plot(df_subset['Period'], p(mdates.date2num(df_subset['Period'])),
                    linestyle='--', label=f'{user_type} Trend')

        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)

        # Set x-axis formatting
        ax = plt.gca()
        if plot_period == 'M':
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.xticks(rotation=45)
        plt.legend(title='User Type')
        plt.tight_layout()
        plt.savefig(plot_save_path, format='png', dpi=300)
        print(f"✅ Plot saved to {plot_save_path}")

    @staticmethod
    @staticmethod
    def plot_humor_2(csv_bot1,
                csv_human1,
                plot_save_path,
                user_types, 
                plot_period='M', 
                target_col_name='Humor',
                plot_title='Bot vs Human Humor Over Time',
                x_label='Time (Monthly)',
                y_label='Average Accuracy Score'):

        # Load CSVs
        dfs = [pd.read_csv(path) for path in [csv_bot1, csv_human1]]

        # Parse datetime and remove timezone
        for df in dfs:
            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce').dt.tz_localize(None)

        # Concatenate datasets
        df_bot = dfs[0][['Created At', target_col_name]].dropna()
        df_human = dfs[1][['Created At', target_col_name]].dropna()

        print(df_bot.head)
        print(df_human.head)

        df_bot = df_bot[df_bot['Created At'].dt.year > 2008]
        df_human = df_human[df_human['Created At'].dt.year > 2008]

        # Label type
        df_bot['Type'] = user_types[0]
        df_human['Type'] = user_types[1]

        # Combine all
        df_all = pd.concat([df_bot, df_human], ignore_index=True)

        # Define period
        period_format = '%Y-%m' if plot_period == 'M' else '%Y'
        df_all['Period'] = df_all['Created At'].dt.to_period(plot_period).dt.to_timestamp()

        # Calculate rolling average
        window_size = 6
        df_all[target_col_name] = df_all.groupby('Type')[target_col_name].transform(
            lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean()
        )

        # Plot
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_all, x='Period', y=target_col_name, hue='Type', errorbar=('ci', 95), marker='.')

        # Add linear trend lines
        for user_type in df_all['Type'].unique():
            df_subset = df_all[df_all['Type'] == user_type]
            z = np.polyfit(mdates.date2num(df_subset['Period']), df_subset[target_col_name], 1)
            p = np.poly1d(z)
            plt.plot(df_subset['Period'], p(mdates.date2num(df_subset['Period'])),
                    linestyle='--', label=f'{user_type} Trend')

        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)

        # Set x-axis formatting
        ax = plt.gca()
        if plot_period == 'M':
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.xticks(rotation=45)
        plt.legend(title='User Type')
        plt.tight_layout()
        plt.savefig(plot_save_path, format='png', dpi=300)
        print(f"✅ Plot saved to {plot_save_path}")

    @staticmethod
    def plot_humor_bot_only(csv_path,  
                plot_save_path, 
                plot_period='M', 
                target_col_name='Humor',
                plot_title='Bot vs Human Grammar Accuracy Over Time',
                x_label='Time (Monthly)',
                y_label='Average Accuracy Score'):

        # Load CSVs
        df = pd.read_csv(csv_path) 

        # Parse datetime and remove timezone
        df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce').dt.tz_localize(None)

        df = df[df['Created At'].dt.year > 2008]

        # Concatenate datasets
        df_bot = df[['Created At', target_col_name]].dropna()
        # df_human = pd.concat([dfs[3], dfs[4], dfs[5]], ignore_index=True)[['Created At', target_col_name]].dropna()

        # Label type
        df_bot['Type'] = 'Bot'
        # df_human['Type'] = 'Human'

        # Combine all
        df_all = df_bot # pd.concat([df_bot, df_human], ignore_index=True)

        # Define period
        period_format = '%Y-%m' if plot_period == 'M' else '%Y'
        df_all['Period'] = df_all['Created At'].dt.to_period(plot_period).dt.to_timestamp()

        # Calculate rolling average
        window_size = 6
        df_all[target_col_name] = df_all.groupby('Type')[target_col_name].transform(
            lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean()
        )

        # Plot
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_all, x='Period', y=target_col_name, hue='Type', errorbar=('ci', 95), marker='.')

        # Add linear trend lines
        for user_type in df_all['Type'].unique():
            df_subset = df_all[df_all['Type'] == user_type]
            z = np.polyfit(mdates.date2num(df_subset['Period']), df_subset[target_col_name], 1)
            p = np.poly1d(z)
            plt.plot(df_subset['Period'], p(mdates.date2num(df_subset['Period'])),
                    linestyle='--', label=f'{user_type} Trend')

        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)

        # Set x-axis formatting
        ax = plt.gca()
        if plot_period == 'M':
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.xticks(rotation=45)
        plt.legend(title='User Type')
        plt.tight_layout()
        plt.savefig(plot_save_path, format='png', dpi=300)
        print(f"✅ Plot saved to {plot_save_path}")
