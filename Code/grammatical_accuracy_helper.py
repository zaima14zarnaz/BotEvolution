import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

class GrammaticalAccuracyHelper:
    def __init__(self):
        pass

    # Correction function
    def correct_grammar(self, text, tokenizer, model):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
        corrected_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return corrected_text

    # def grammatical_accuracy(self, text, tokenizer, model, device):
    #     # inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    #     # outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    #     # corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     tokenized_sentence = tokenizer('gec: ' + text, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    #     corrected_sentence = tokenizer.decode(
    #         model.generate(
    #             input_ids = tokenized_sentence.input_ids,
    #             attention_mask = tokenized_sentence.attention_mask, 
    #             max_length=128,
    #             num_beams=5,
    #             early_stopping=True,
    #         )[0],
    #         skip_special_tokens=True, 
    #         clean_up_tokenization_spaces=True
    #     )

    #     # Compare original and corrected text to compute accuracy
    #     orig_tokens = text.split()
    #     corrected_tokens = corrected_sentence.split()

    #     differences = sum(1 for o, c in zip(orig_tokens, corrected_tokens) if o != c)
    #     differences += abs(len(orig_tokens) - len(corrected_tokens))

    #     accuracy = (1 - differences / max(len(orig_tokens), 1)) * 100
    #     return round(accuracy, 2)

    def grammatical_accuracy(self, text, tokenizer, model, device):
        # inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        # outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
        # corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Tokenize input
        tokenized_sentence = tokenizer(
            'gec: ' + text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Move tensors to the same device as model
        input_ids = tokenized_sentence.input_ids.to(device)
        attention_mask = tokenized_sentence.attention_mask.to(device)

        # Generate corrected sentence
        corrected_sentence = tokenizer.decode(
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )[0],
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )

        # Compare tokens
        orig_tokens = text.split()
        corrected_tokens = corrected_sentence.split()

        differences = sum(1 for o, c in zip(orig_tokens, corrected_tokens) if o != c)
        differences += abs(len(orig_tokens) - len(corrected_tokens))

        accuracy = (1 - differences / max(len(orig_tokens), 1)) * 100
        return round(accuracy, 2)



    def plot_grammatical_accuracy_kaggle(self, csv_path, plot_save_path):
        # Load the CSV
        df = pd.read_csv(csv_path)

        # Convert 'Created At' to datetime
        df['Created At'] = pd.to_datetime(df['Created At'], format='%Y-%m-%d %H:%M:%S')

        # Create 'Year-Month' column for grouping
        df['Year-Month'] = df['Created At'].dt.to_period('M').astype(str)
        print(df['Year-Month'].unique())

        # Group by 'Year-Month' and compute average grammatical accuracy
        monthly_avg = df.groupby('Year-Month')['Grammatical Accuracy'].mean().reset_index()
        print(monthly_avg)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_avg['Year-Month'], monthly_avg['Grammatical Accuracy'], marker='o')
        plt.xticks(rotation=45)
        plt.xlabel('Year-Month')
        plt.ylabel('Average Grammatical Accuracy')
        plt.title('Monthly Average Grammatical Accuracy Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plot.png', format='png', dpi=300)  # or 'plot.jpeg', format='jpeg' # plt.show()

    @staticmethod
    def plot_grammatical_accuracy_1(csv_bot1,
                csv_bot2, 
                csv_human1,
                csv_human2,
                plot_save_path,
                user_types, 
                plot_period='M', 
                target_col_name='Grammatical Accuracy',
                plot_title='Bot vs Human Grammar Accuracy Over Time',
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
    def plot_grammatical_accuracy_2(csv_bot1,
                csv_human1,
                plot_save_path,
                user_types, 
                plot_period='M', 
                target_col_name='Grammatical Accuracy',
                plot_title='Bot vs Human Grammar Accuracy Over Time',
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
    def plot_grammatical_accuracy_3(csv_bot_1,
                                    csv_bot_2,
                                    csv_llm, 
                plot_save_path,
                user_types,  
                plot_period='M', 
                target_col_name='Grammatical Accuracy',
                plot_title='Bot vs Human Grammar Accuracy Over Time',
                x_label='Time (Monthly)',
                y_label='Average Accuracy Score'):

        dfs = [pd.read_csv(path) for path in [csv_bot_1, csv_bot_2, csv_llm]]

        # Parse datetime and remove timezone
        for df in dfs:
            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce').dt.tz_localize(None)

        # Concatenate datasets
        df_bot = pd.concat([dfs[0],dfs[1]],ignore_index=True)[['Created At', target_col_name]].dropna()
        df_human = dfs[2][['Created At', target_col_name]].dropna()
        
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
    def plot_grammatical_accuracy_1(csv_bot1,
                csv_bot2, 
                csv_human1,
                csv_human2,
                plot_save_path,
                user_types, 
                plot_period='M', 
                target_col_name='Grammatical Accuracy',
                plot_title='Bot vs Human Grammar Accuracy Over Time',
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