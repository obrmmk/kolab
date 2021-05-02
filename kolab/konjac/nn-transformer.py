import pickle
import random
import numpy as np

from torch.utils.data import Dataset

import math
from einops import rearrange

import torch
from torch import nn

from tqdm import tqdm
import random
from collections import Counter
import pickle
from pathlib import Path

import spacy

import click
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from torch.optim import Adam
from Optim import ScheduledOptim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ParallelLanguageDataset
from model import LanguageTransformer

import pickle
from einops import rearrange

import spacy
import torch


def forward_model(model, src, tgt):
    src = torch.tensor(src).unsqueeze(0).long().to('cuda')
    tgt = torch.tensor(tgt).unsqueeze(0).to('cuda')
    tgt_mask = gen_nopeek_mask(tgt.shape[1]).to('cuda')
    output = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=tgt_mask)

    return output.squeeze(0).to('cpu')


def tokenize(sentence, freq_list, lang_model):
    punctuation = ['(', ')', ':', '"', ' ']

    sentence = sentence.lower()
    sentence = [tok.text for tok in lang_model.tokenizer(sentence) if tok.text not in punctuation]
    return [freq_list[word] if word in freq_list else freq_list['[OOV]'] for word in sentence]


def detokenize(sentence, freq_list):
    freq_list = {v: k for k, v in freq_list.items()}
    return [freq_list[token] for token in sentence]


def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask



def train(train_loader, valid_loader, model, optim, criterion, num_epochs):
    print_every = 500
    model.train()

    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(total=print_every, leave=False)
        total_loss = 0

        # Shuffle batches every epoch
        train_loader.dataset.shuffle_batches()
        for step, (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in enumerate(iter(train_loader)):
            total_step += 1

            # Send the batches and key_padding_masks to gpu
            src, src_key_padding_mask = src[0].to('cuda'), src_key_padding_mask[0].to('cuda')
            tgt, tgt_key_padding_mask = tgt[0].to('cuda'), tgt_key_padding_mask[0].to('cuda')
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
            tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to('cuda')

            # Forward
            optim.zero_grad()
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            # Backpropagate and update optim
            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            pbar.update(1)
            if step % print_every == print_every - 1:
                pbar.close()
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t '
                      f'Train Loss: {total_loss / print_every}')
                total_loss = 0

                pbar = tqdm(total=print_every, leave=False)

        # Validate every epoch
        pbar.close()
        val_loss = validate(valid_loader, model, criterion)
        val_losses.append((total_step, val_loss))
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model, 'output/transformer.pth')
        print(f'Val Loss: {val_loss}')
    return train_losses, val_losses


def validate(valid_loader, model, criterion):
    pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    total_loss = 0
    for src, src_key_padding_mask, tgt, tgt_key_padding_mask in iter(valid_loader):
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to('cuda'), src_key_padding_mask[0].to('cuda')
            tgt, tgt_key_padding_mask = tgt[0].to('cuda'), tgt_key_padding_mask[0].to('cuda')
            memory_key_padding_mask = src_key_padding_mask.clone()
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:].contiguous()
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to('cuda')

            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            total_loss += loss.item()
            pbar.update(1)

    pbar.close()
    model.train()
    return total_loss / len(valid_loader)


def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sentence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def process_lang_data(data_path, lang, punctuation, train_indices, val_indices, test_indices):
    lang_data = load_data(data_path)
    lang_model = spacy.load(lang, disable=['tagger', 'parser', 'ner'])

    # Tokenize the sentences
    processed_sentences = [process_sentences(lang_model, sentence, punctuation) for sentence in tqdm(lang_data)]

    train = [processed_sentences[i] for i in train_indices]

    # Get the 10000 most common tokens
    freq_list = Counter()
    for sentence in train:
        freq_list.update(sentence)
    freq_list = freq_list.most_common(10000)

    # Map words in the dictionary to indices but reserve 0 for padding,
    # 1 for out of vocabulary words, 2 for start-of-sentence and 3 for end-of-sentence
    freq_list = {freq[0]: i + 4 for i, freq in enumerate(freq_list)}
    freq_list['[PAD]'] = 0
    freq_list['[OOV]'] = 1
    freq_list['[SOS]'] = 2
    freq_list['[EOS]'] = 3
    processed_sentences = [map_words(sentence, freq_list) for sentence in tqdm(processed_sentences)]

    # Split the data
    train = [processed_sentences[i] for i in train_indices]
    val = [processed_sentences[i] for i in val_indices]
    test = [processed_sentences[i] for i in test_indices]

    # Save the data
    with open(f'data/processed/{lang}/train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'data/processed/{lang}/val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(f'data/processed/{lang}/test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open(f'data/processed/{lang}/freq_list.pkl', 'wb') as f:
        pickle.dump(freq_list, f)


def process_sentences(lang_model, sentence, punctuation):
    """
     Processes sentences by lowercasing text, ignoring punctuation, and using Spacy tokenization
             Parameters:
                     lang_model: Spacy language model
                     sentence (str): Sentence to be tokenized
                     punctuation (arr): Array of punctuation to be ignored
             Returns:
                     sentence (arr): Tokenized sentence
     """
    sentence = sentence.lower()
    sentence = [tok.text for tok in lang_model.tokenizer(sentence) if tok.text not in punctuation]

    return sentence


def load_data(data_path):
    data = []
    with open(data_path) as fp:
        for line in fp:
            data.append(line.strip())
    return data


def map_words(sentence, freq_list):
    return [freq_list[word] for word in sentence if word in freq_list]


def generate_indices(data_len):
    """
     Generate train, validation, and test indices
             Parameters:
                     data_len (int): Amount of sentences in the dataset
             Returns:
                     train_indices (arr): Array of indices for train dataset
                     val_indices (arr): Array of indices for validation dataset
                     test_indices (arr): Array of indices for test dataset
     """
    indices = [i for i in range(data_len)]
    random.shuffle(indices)

    # 80:20:0 train validation test split
    train_idx = int(data_len * 0.8)
    val_idx = train_idx + int(data_len * 0.2)
    return indices[:train_idx], indices[train_idx:val_idx], indices[val_idx:]



# From https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr



class LanguageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        """
        Initializes the model
                Parameters:
                        vocab_size (int): The amount of tokens in both vocabularies (including start, end, etc tokens)
                        d_model (int): Expected number of features in the encoder/decoder inputs, also used in embeddings
                        nhead (int): Number of heads in the transformer
                        num_encoder_layers (int): Number of sub-encoder layers in the transformer
                        num_decoder_layers (int): Number of sub-decoder layers in the transformer
                        dim_feedforward (int): Dimension of the feedforward network in the transformer
                        max_seq_length (int): Maximum length of each tokenized sentence
                        pos_dropout (float): Dropout value in the positional encoding
                        trans_dropout (float): Dropout value in the transformer
        """
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(vocab_size, d_model)
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask):
        # Reverse the shape of the batches from (num_sentences, num_tokens_in_each_sentence)
        src = rearrange(src, 'n s -> s n')
        tgt = rearrange(tgt, 'n t -> t n')

        # Embed the batches, scale by sqrt(d_model), and add the positional encoding
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        # Send the batches to the model
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        # Rearrange to batch-first
        output = rearrange(output, 't n e -> n t e')

        # Run the output through an fc layer to return values for each token in the vocab
        return self.fc(output)


# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ParallelLanguageDataset(Dataset):
    def __init__(self, data_path_1, data_path_2, num_tokens, max_seq_length):
        """
        Initializes the dataset
                Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                        max_seq_length (int): Maximum number of tokens in each sentence pair
        """
        self.num_tokens = num_tokens
        self.data_1, self.data_2, self.data_lengths = load_data(data_path_1, data_path_2, max_seq_length)

        self.batches = gen_batches(num_tokens, self.data_lengths)

    def __getitem__(self, idx):
        src, src_mask = getitem(idx, self.data_1, self.batches, True)
        tgt, tgt_mask = getitem(idx, self.data_2, self.batches, False)

        return src, src_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):
        self.batches = gen_batches(self.num_tokens, self.data_lengths)


def gen_batches(num_tokens, data_lengths):
    """
     Returns the batched data
             Parameters:
                     num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                     data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
                                         and values of the indices that correspond to these parallel sentences
             Returns:
                     batches (arr): List of each batch (which consists of an array of indices)
     """

    # Shuffle all the indices
    for k, v in data_lengths.items():
        random.shuffle(v)

    batches = []
    prev_tokens_in_batch = 1e10
    for k in sorted(data_lengths):
        # v contains indices of the sentences
        v = data_lengths[k]
        total_tokens = (k[0] + k[1]) * len(v)

        # Repeat until all the sentences in this key-value pair are in a batch
        while total_tokens > 0:
            tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k[0] + k[1])
            sentences_in_batch = tokens_in_batch // (k[0] + k[1])

            # Combine with previous batch if it can fit
            if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
                batches[-1].extend(v[:sentences_in_batch])
                prev_tokens_in_batch += tokens_in_batch
            else:
                batches.append(v[:sentences_in_batch])
                prev_tokens_in_batch = tokens_in_batch
            # Remove indices from v that have been added in a batch
            v = v[sentences_in_batch:]

            total_tokens = (k[0] + k[1]) * len(v)
    return batches


def load_data(data_path_1, data_path_2, max_seq_length):
    """
    Loads the pickle files created in preprocess-data.py
            Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        max_seq_length (int): Maximum number of tokens in each sentence pair

            Returns:
                    data_1 (arr): Array of tokenized English sentences
                    data_2 (arr): Array of tokenized French sentences
                    data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
                                         and values of the indices that correspond to these parallel sentences
    """
    with open(data_path_1, 'rb') as f:
        data_1 = pickle.load(f)
    with open(data_path_2, 'rb') as f:
        data_2 = pickle.load(f)

    data_lengths = {}
    for i, (str_1, str_2) in enumerate(zip(data_1, data_2)):
        if 0 < len(str_1) <= max_seq_length and 0 < len(str_2) <= max_seq_length - 2:
            if (len(str_1), len(str_2)) in data_lengths:
                data_lengths[(len(str_1), len(str_2))].append(i)
            else:
                data_lengths[(len(str_1), len(str_2))] = [i]
    return data_1, data_2, data_lengths


def getitem(idx, data, batches, src):
    """
    Retrieves a batch given an index
            Parameters:
                        idx (int): Index of the batch
                        data (arr): Array of tokenized sentences
                        batches (arr): List of each batch (which consists of an array of indices)
                        src (bool): True if the language is the source language, False if it's the target language

            Returns:
                    batch (arr): Array of tokenized English sentences, of size (num_sentences, num_tokens_in_sentence)
                    masks (arr): key_padding_masks for the sentences, of size (num_sentences, num_tokens_in_sentence)
    """

    sentence_indices = batches[idx]
    if src:
        batch = [data[i] for i in sentence_indices]
    else:
        # If it's in the target language, add [SOS] and [EOS] tokens
        batch = [[2] + data[i] + [3] for i in sentence_indices]

    # Get the maximum sentence length
    seq_length = 0
    for sentence in batch:
        if len(sentence) > seq_length:
            seq_length = len(sentence)

    masks = []
    for i, sentence in enumerate(batch):
        # Generate the masks for each sentence, False if there's a token, True if there's padding
        masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
        # Add 0 padding
        batch[i] = sentence + [0 for _ in range(seq_length - len(sentence))]

    return np.array(batch), np.array(masks)
