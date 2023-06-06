import csv
import torch

class DataCollatorCustom:
    def __init__(self,
                filename,
                tokenizer,
                max_length=128,
                ):
        self.tokenizer = tokenizer
        self.filename = filename
        self.max_length = max_length

    def load_dataset(self):
        filename = self.filename

        tokens_list, labels_list = [], []
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for corrupted, sentence in reader:
                tokens, labels = self.formatting(corrupted, sentence)
                tokens_list.append(tokens)
                labels_list.append(labels)

        sentences = [self.tokenizer.decode(tokens) for tokens in tokens_list]
        
        encodings = self.tokenizer(
            sentences, return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_length)
        
        encodings["labels"] = torch.tensor(labels_list)
        return encodings

    def formatting(self, input_text, target_text):
        input_tokens = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True)
        target_tokens = self.tokenizer.encode(target_text, max_length=self.max_length, truncation=True)

        tokens = [self.tokenizer.bos_token_id] + input_tokens \
            + [self.tokenizer.sep_token_id] + target_tokens \
            + [self.tokenizer.eos_token_id]

        labels = [-100] * (len(input_tokens) + 2) \
            + target_tokens + [self.tokenizer.eos_token_id] \
            + [-100] * (self.max_length - len(tokens))
        labels = labels[:self.max_length]
        
        return tokens, labels