# Importing modules
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import XLNetConfig, XLNetLMHeadModel, XLNetModel, XLNetTokenizer
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from transformers import PreTrainedModel, PreTrainedTokenizer , BertPreTrainedModel

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import codecs
from torch.nn.utils.rnn import pack_padded_sequence
import os

from reader import *
from config import *
from model import *
from eval_metric import *

# Importing the tokenizer for Transformer model
tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case = False)

# test_out
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)

# Tokenization for train, dev and test data
f_tokenized_texts, f_token_map, f_sent_length = read_test_token_map(test_file)
f_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in f_tokenized_texts]

# Pad the input tokens
f_input_ids = pad_sequences(f_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
f_token_map = pad_sequences(f_token_map, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create a mask of 1s for each token followed by 0s for padding
f_attention_masks = []
for seq in f_input_ids:
  seq_mask = [float(i>0) for i in seq]
  f_attention_masks.append(seq_mask)

# Converting to Tensors
f_input_ids = torch.tensor(f_input_ids)
f_token_map = torch.tensor(f_token_map )
f_attention_masks = torch.tensor(f_attention_masks)
f_sent_length = torch.tensor(f_sent_length)

# Create an iterator of our data with torch DataLoader 
test_data = TensorDataset(f_input_ids, f_token_map, f_attention_masks, f_sent_length)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,shuffle = False)

test_words, test_word_ids = read_for_output(test_file)


model = transformer_model(model_name)
model.load_state_dict(torch.load('mode.pth'))
model.to(device)

def get_pred(sample):
  print("")
  print("Running test...")

  model.eval()
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  iii = 0

  s = ""
  sentence_id = ""
    
  # Add sample to device
  sample = tuple(t.to(device) for t in sample)

  # Unpack the inputs from our dataloader
  v_input_ids = sample[0].to(device)
  v_input_mask = sample[2].to(device)
  v_token_starts = sample[1].to(device)
  v_sent_length = sample[3]
    
  # Telling the model not to compute or store gradients, saving memory and speeding up
  with torch.no_grad():        
      output = model(
      v_input_ids.to(torch.int64),
      v_input_mask.to(torch.int64),
      v_token_starts.to(torch.int64),
      v_sent_length.to(torch.int64)
      )

  pred_labels = output[1]

  pred_labels = pred_labels.detach().cpu().numpy()

  for i in range(v_input_ids.size()[0]):
    for j in range(len(test_words[iii])):
      if sentence_id == iii:
        s = s + "{}\t{}\t{}\t".format(test_word_ids[iii][j], test_words[iii][j], pred_labels[i][j]) + "\n"
      else:
        s = s + "\n" + "{}\t{}\t{}\t".format(test_word_ids[iii][j], test_words[iii][j], pred_labels[i][j]) + "\n"
        sentence_id = iii
    iii = iii + 1
  s = s +"\n"
  print(s)
  return s

for n, batch in enumerate(test_dataloader):
    print(batch)
    if n == 10:
        sample = batch
        break
print(len(sample))
get_pred(sample)