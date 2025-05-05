from dataset import DatasetHandler
from humor_detection import HumorDetector

dataset_addr = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/LLM_tweets/cleaned_batch_3001_4700.csv"
analysis_csv = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/llm_tweets_analysis_set3.csv"# ['User ID', 'Username', 'Tweet', 'Created At', 'Year', 'Grammatical Accuracy', 'Sentence Complexity']
# grammatical_accuracy_plot_path = "/home/zaimaz/Desktop/research1/BotEvolution/Dataset/grammatical_accuracy_plot_cresci_human_year.jpeg"
# dataset = DatasetHandler(analysis_csv)
# dataset.view_data

skiprows = 0
nrows = 2000000

# dataset = DatasetHandler(dataset_addr)
main_dataset = DatasetHandler(dataset_addr, encoding='ISO-8859-1')
analysis_dataset = DatasetHandler(analysis_csv, encoding='ISO-8859-1')
text_analyzer = HumorDetector(main_dataset.df, analysis_dataset.df)

text_analyzer.analyze_humor(analysis_csv, tweet_col_name='humanized_tweet', start_from=skiprows)
# DatasetHandler.write_to_csv(df, analysis_csv)
# print(text_analyzer.grammatical_accuracy("", text_analyzer.tokenizer, text_analyzer.model, text_analyzer.device))