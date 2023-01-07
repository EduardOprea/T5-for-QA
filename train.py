from __future__ import print_function
from typing import List, Tuple
from tqdm import tqdm
import torch

from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader
import argparse
from birthplace_dataset import BirthPlaceDataset 


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training T5 T2T model')

    parser.add_argument('--t5_model', type=str, default="t5-small",
                        help="What type of T5 model do you want use?")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs (default: 40)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (Adam) (default: 1e-4)')

    parser.add_argument('--workers', type=int, default=1,
                        help='number of working units used to load the data (default: 1)')

    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)')

    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Frequency of checkpointing')


    parser.add_argument("--questions_file_val", type=str, default = "E:\Fisierele mele\Facultate\DNN\HW3\\birth_places_test.tsv")
    parser.add_argument("--questions_file_train", type=str, default = "E:\Fisierele mele\Facultate\DNN\HW3\\birth_places_train.tsv")
    parser.add_argument("--context_file", type=str, default = "E:\Fisierele mele\Facultate\DNN\HW3\\wiki.txt")
    parser.add_argument("--results_path", type=str, default = "results")
                        

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def train(args, model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizer, optimizer: AdamW,
 train_set: BirthPlaceDataset, validation_set: BirthPlaceDataset, num_train_epochs: int,
  device: str, batch_size: int, max_input_length: int = 512):
    print("Training arguments ", args)
    my_trainset_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                        num_workers=args.workers)
    my_validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size,
                                          num_workers=args.workers)


    ckpt_path = f"{args.results_path}/{model.name_or_path}"
    print("Model checkpoints are saved in ", ckpt_path)
    # set training mode on the model
    model.train()

    # model to device
    model.to(device)

    accuracy_old = 0.0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        for contexts,questions,answers in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(questions,contexts)))
            encoded_inputs = tokenizer(
                                    inputs,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
            encoded_targets = tokenizer(
                                    answers,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )

            input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
            encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

            input_ids = input_ids.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        model.eval()
        with torch.no_grad():
            model_predictions_encoded = []
            target_encoded = []
            for contexts, questions, answers in tqdm(my_validation_dataloader):
                inputs = list(map(lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}", zip(
                    questions, contexts)))
                encoded_inputs = tokenizer(
                    inputs,
                    padding="longest",
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_targets = tokenizer(
                    answers,
                    padding="longest",
                    max_length=max_input_length,
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
        accuracy = validation_set.evaluate(
            target_encoded, model_predictions_encoded)
        print(f"Accuracy on the validation set is ", accuracy)

        if accuracy > accuracy_old :
            model.save_pretrained(f'{ckpt_path}/model/best-acc')
            tokenizer.save_pretrained(f'{ckpt_path}/tokenizer/best-acc')
            accuracy_old = accuracy
        if epoch + 1 % args.checkpoint_freq == 0:
            model.save_pretrained(f'{ckpt_path}/model/checkpoint-{epoch+1}')
            tokenizer.save_pretrained(f'{ckpt_path}/tokenizer/checkpoint-{epoch+1}')
        model.train()

    model.save_pretrained(
        f'{ckpt_path}/model/checkpoint-{epoch+1}')
    tokenizer.save_pretrained(
        f'{ckpt_path}/tokenizer/checkpoint-{epoch+1}')


if __name__ == '__main__':
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    # Set seed
    set_seed(args.seed)

    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    # creating the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_set = BirthPlaceDataset(args.questions_file_train, args.context_file, tokenizer)
    validation_set = BirthPlaceDataset(args.questions_file_val, args.context_file, tokenizer)

    train(args,
          model=model,
          tokenizer=tokenizer,
          optimizer=optimizer,
          train_set=train_set,
          validation_set=validation_set,
          num_train_epochs=args.epochs, device=args.device, batch_size=args.batch_size)