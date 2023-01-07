import torch
import pandas as pd
from tqdm import tqdm

class BirthPlaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, context_file, tokenizer):
        self.tokenizer = tokenizer
        
        self.df = pd.read_table(dataset_path) 

        with open(context_file, encoding="utf-8") as f:
            context_lines = f.readlines()

        context_lines = [line.rstrip("\n") for line in context_lines]

        self.df['Person'] = self.df.apply(lambda row: self.get_person_name(row), axis = 1)
        self.df['Context'] = self.df.apply(lambda row: self.get_context(row, context_lines), axis = 1)
        print(f"Dataframe after computing context : ", self.df.head())
        self.length = len(self.df.index)
        print(f"Dataframe length : ", self.length)

    def __len__(self):
        return self.length
    def __getitem__(self, index: int):
        return self.df.iloc[index]['Context'], self.df.iloc[index]['Question'], self.df.iloc[index]['Answer']
    def get_person_name(self, row):
        start = "Where was "
        end = " born?" 
        question = row['Question']
        startIndex = question.find(start)
        endIndex = question.rfind(end)
        if startIndex == -1 or endIndex == -1:
            raise IndexError("The question doesn't have the expected format -- ", question)

        return question[startIndex + len(start) : endIndex]

    def get_context(self, row, context_lines):
        name = row['Person']
        context = next((context for context in context_lines if name in context), None)
        if context is None:
            raise ValueError(f"Context for person {name} can not be retrieved from wiki.txt")
        return context

    def __exact_match_score(self, prediction, ground_truth):
        """_summary_
        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_
        Returns:
            _type_: _description_
        """
        if len(ground_truth) == len(prediction):
            if all(token1 == token2 for token1, token2 in zip(ground_truth,prediction)):
                return 1
        return 0

    def evaluate(self, predictions, gold_answers):
        """_summary_
        Args:
            predictions (_type_): _description_
            gold_answers (_type_): _description_
        Returns:
            _type_: _description_
        """
        exact_match = 0

        for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
            # Remove pad token
            tokens_to_remove = {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id
            }
            prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
            ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
            exact_match += self.__exact_match_score(prediction, ground_truths)
        return 100*exact_match/len(predictions)