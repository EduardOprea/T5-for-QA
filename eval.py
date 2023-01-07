import argparse
from birthplace_dataset import BirthPlaceDataset
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for evaluating T5 T2T model')

    parser.add_argument('--t5_model', type=str, default="t5-small",
                        help="What type of T5 model do you want use?")

    parser.add_argument('--dataset', type=str, default='duorc-SelfRC',
                        help="Dataset to be used, if more level provided for the dataset use the '-' token, e.g. duorc-SelfRC")
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='mini-batch size (default: 1)')
    
    parser.add_argument('--workers', type=int, default=1,
                        help='number of working units used to load the data (default: 10)')

    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)')
    parser.add_argument("--questions_file_test", type=str, default = "E:\Fisierele mele\Facultate\DNN\HW3\\birth_places_test.tsv")
    parser.add_argument("--context_file", type=str, default = "E:\Fisierele mele\Facultate\DNN\HW3\\wiki.txt")
    parsed_arguments = parser.parse_args()

    return parsed_arguments

if __name__ == '__main__':
    args = parse_command_line_arguments()
    print("Eval arguments ", args)
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

    test_set = BirthPlaceDataset(args.questions_file_test, args.context_file, tokenizer)
    test_loader = DataLoader(test_set, args.batch_size)
    
    device = args.device
    model.to(device)

    model.eval()
    with torch.no_grad():
        model_predictions_encoded = []
        target_encoded = []
        for contexts, questions, answers in tqdm(test_loader):
            inputs = list(map(lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}", zip(
                questions, contexts)))
            encoded_inputs = tokenizer(
                inputs,
                padding="longest",
                max_length=args.max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = tokenizer(
                    answers,
                    padding="longest",
                    max_length=args.max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
            encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids
            
            encoded_inputs = encoded_inputs.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)
            model_predictions = model.generate(
                input_ids=encoded_inputs, attention_mask=attention_mask)

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()

        exact_match = test_set.evaluate(
            target_encoded, model_predictions_encoded)

        print(f"Accuracy on the test set is: {exact_match:.2f}")