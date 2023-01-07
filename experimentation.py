import pandas as pd
import re

def get_person_name(row):
    start = "Where was "
    end = " born?" 
    question = row['Question']
    startIndex = question.find(start)
    endIndex = question.rfind(end)
    if startIndex == -1 or endIndex == -1:
        raise IndexError("The question doesn't have the expected format -- ", question)

    return question[startIndex + len(start) : endIndex]

def get_context(row, context_lines):
    name = row['Person']
    context = next((context for context in context_lines if name in context), None)
    if context is None:
        print("stop right here chief")
    return context

test_file = "E:\Fisierele mele\Facultate\DNN\HW3\\birth_places_train.tsv"
context_file = "E:\Fisierele mele\Facultate\DNN\HW3\\wiki.txt"
with open(context_file, encoding="utf-8") as f:
    context_lines = f.readlines()

df = pd.read_table(test_file) 
df['Person'] = df.apply(lambda row: get_person_name(row), axis = 1)

df['Context'] = df.apply(lambda row: get_context(row, context_lines), axis = 1)

print(df.head())





