from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from grammatical_accuracy_helper import GrammaticalAccuracyHelper
from dataset import DatasetHandler
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time


class GrammaticalAccuracy(GrammaticalAccuracyHelper):
    def __init__(self, main_df, output_df):
        super().__init__()
        # Load model and tokenizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1").to(self.device)
        self.model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.analysis_csv_df = output_df
        self.main_df = main_df

    def analyze_grammatical_accuracy(self, csv_addr, tweet_col_name, start_from):
        texts = self.main_df[tweet_col_name][start_from:]
        startTime = time.time()
        endTime = 0
        for id, text in enumerate(texts):
            index = start_from + id
            text = str(text)
            self.analysis_csv_df.loc[index, 'Grammatical Accuracy'] = self.grammatical_accuracy(text, self.tokenizer, self.model, self.device)
            if index % 100 == 0 and index != 0:
                DatasetHandler.write_to_csv(self.analysis_csv_df, csv_addr)

                endTime = time.time()
                print(f'Saved upto {index} datapoints in {round((endTime - startTime),2)}', flush=True)

                startTime = time.time()
                
    
        return self.analysis_csv_df
    
            

