from dataset import DatasetHandler
from humor_detection import HumorDetector

# dataset_addr = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/tweets.csv"
analysis_csv_cresci_bot = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci_data_analysis_bots_social_spambots_3.csv"
analysis_csv_cresci_human = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci_data_analysis_human.csv"
analysis_csv_kaggle_bot = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/kaggle_data_analysis_bots.csv"
analysis_csv_kaggle_human = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/kaggle_data_analysis_humans.csv" # ['User ID', 'Username', 'Tweet', 'Created At', 'Year', 'Grammatical Accuracy', 'Sentence Complexity']
analysis_csv_twibot_bot = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/twibot_data_analysis_bots.csv"
analysis_csv_twibot_human = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/twibot_data_analysis_humans.csv"
analysis_csv_llama = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/llm_tweets_analysis_merged.csv"
humor_plot_path = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/bot_vs_human_humor_twibot.jpeg"

# main_dataset = DatasetHandler(dataset_addr, encoding='ISO-8859-1')
# analysis_dataset = DatasetHandler(analysis_csv_cresci_bot, analysis_csv_kaggle_bot, analysis_csv_cresci_human, analysis_csv_kaggle_human, encoding='ISO-8859-1')
# text_analyzer = TextAnalyzer(main_dataset.df, analysis_dataset.df)

# HumorDetector.plot_humor_1(analysis_csv_twibot_bot,
#                          analysis_csv_cresci_bot, 
#                          analysis_csv_twibot_human,
#                          analysis_csv_cresci_human,
#                          humor_plot_path, 
#                          plot_period='M',
#                          plot_title='Evolution of Bot vs Human Humor Over Time', 
#                          x_label='Time(Monthly)', 
#                          y_label='Mean Humor',
#                          user_types=['Bot', 'Human'])

HumorDetector.plot_humor_2(analysis_csv_twibot_bot,
                         analysis_csv_twibot_human,
                         humor_plot_path, 
                         plot_period='M',
                         plot_title='Evolution of Bot vs Human Humor Over Time', 
                         x_label='Time(Monthly)', 
                         y_label='Mean Humor',
                         user_types=['Human', 'Bot'])

# HumorDetector.plot_humor_bot_only(analysis_csv_twibot_bot, 
#                                   humor_plot_path, 
#                                   plot_period='M',
#                                   plot_title='Evolution of Bot Humor Over Time', 
#                                   x_label='Time(Monthly)', 
#                                   y_label='Mean Humor')

