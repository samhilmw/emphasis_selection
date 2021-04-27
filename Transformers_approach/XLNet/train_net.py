# Redirecting Deprication Warnings to /dev/null
import sys, platform

sv_std = sys.stderr
os_name = platform.system()

if os_name == "Windows":
    f = open('nul', 'w')
elif os_name == "Linux":
    f = open(os.devnull, 'w')
sys.stderr = f

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

# Restoring sys.stderr
sys.stderr = sv_std

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)

# Tokenization for train, dev and test data
t_tokenized_texts, t_token_map, t_token_label, t_sent_length = read_token_map(train_file)
d_tokenized_texts, d_token_map, d_token_label, d_sent_length = read_token_map(dev_file)
f_tokenized_texts, f_token_map, f_sent_length = read_test_token_map(test_file)

# Converting the tokens to their index numbers
t_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in t_tokenized_texts]
d_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in d_tokenized_texts]
f_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in f_tokenized_texts]

# Pad the input tokens
t_input_ids = pad_sequences(t_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
t_token_map = pad_sequences(t_token_map, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
t_token_label = pad_sequences(t_token_label, maxlen=MAX_LEN, dtype="float", truncating="post", padding="post")

d_input_ids = pad_sequences(d_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
d_token_map = pad_sequences(d_token_map, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
d_token_label = pad_sequences(d_token_label, maxlen=MAX_LEN, dtype="float", truncating="post", padding="post")

f_input_ids = pad_sequences(f_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
f_token_map = pad_sequences(f_token_map, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create a mask of 1s for each token followed by 0s for padding

t_attention_masks = []
for seq in t_input_ids:
  seq_mask = [float(i>0) for i in seq]
  t_attention_masks.append(seq_mask)
print(t_attention_masks[100])

d_attention_masks = []
for seq in d_input_ids:
  seq_mask = [float(i>0) for i in seq]
  d_attention_masks.append(seq_mask)
print(d_attention_masks[0])

f_attention_masks = []
for seq in f_input_ids:
  seq_mask = [float(i>0) for i in seq]
  f_attention_masks.append(seq_mask)
print(f_attention_masks[50])

# Converting to Tensors
t_input_ids = torch.tensor(t_input_ids)
t_token_map = torch.tensor(t_token_map )
t_token_label = torch.tensor(t_token_label)
t_attention_masks = torch.tensor(t_attention_masks)
t_sent_length = torch.tensor(t_sent_length)

d_input_ids = torch.tensor(d_input_ids)
d_token_map = torch.tensor(d_token_map )
d_token_label = torch.tensor(d_token_label)
d_attention_masks = torch.tensor(d_attention_masks)
d_sent_length = torch.tensor(d_sent_length)

f_input_ids = torch.tensor(f_input_ids)
f_token_map = torch.tensor(f_token_map )
f_attention_masks = torch.tensor(f_attention_masks)
f_sent_length = torch.tensor(f_sent_length)


# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(t_input_ids, t_token_map, t_token_label, t_attention_masks, t_sent_length)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(d_input_ids, d_token_map, d_token_label, d_attention_masks, d_sent_length)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, shuffle = False)
test_data = TensorDataset(f_input_ids, f_token_map, f_attention_masks, f_sent_length)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,shuffle = False)

dev_words, dev_word_ids = read_for_output(dev_file)
test_words, test_word_ids = read_for_output(test_file)

def test(model):
  print("")
  print("Running test...")

  model.eval()
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  iii = 0

  s = ""
  sentence_id = ""

  for batch in test_dataloader:
      
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      
      # Unpack the inputs from our dataloader
      v_input_ids = batch[0].to(device)
      v_input_mask = batch[2].to(device)
      v_token_starts = batch[1].to(device)
      v_sent_length = batch[3]
            
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
      
  print("testing complete\n")
  # print(s)
  return s

def validation(model):
  print("")
  print("Running Validation...")

  model.eval()
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  num_m = [0, 0, 0, 0]
  score_m = [0, 0, 0, 0]

  iii = 0

  s = ""
  sentence_id = ""

  for batch in validation_dataloader:
      
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      
      # Unpack the inputs from our dataloader
      v_input_ids = batch[0].to(device)
      v_input_mask = batch[3].to(device)
      v_token_starts = batch[1].to(device)
      v_labels = batch[2].to(device)
      v_sent_length = batch[4]
            
      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():        
          output = model(
            v_input_ids.to(torch.int64),
            v_input_mask.to(torch.int64),
            v_token_starts.to(torch.int64), 
            v_sent_length.to(torch.int64),
            v_labels.to(torch.double)
          )
      pred_labels = output[1]

      pred_labels = pred_labels.detach().cpu().numpy()
      v_labels = v_labels.to('cpu').numpy()

      for i in range(v_input_ids.size()[0]):
        for j in range(len(dev_words[iii])):
          if sentence_id == iii:
            s = s + "{}\t{}\t{}\t{}".format(dev_word_ids[iii][j], dev_words[iii][j], pred_labels[i][j],v_labels[i][j]) + "\n"
          else:
            s = s + "\n" + "{}\t{}\t{}\t{}".format(dev_word_ids[iii][j], dev_words[iii][j], pred_labels[i][j],v_labels[i][j]) + "\n"
            sentence_id = iii
        iii = iii + 1
      s = s +"\n"
      
      pred_labels, v_labels = fix_padding(pred_labels, v_labels, v_sent_length)

      batch_num_m, batch_score_m = match_M(pred_labels, v_labels)
      num_m = [sum(i) for i in zip(num_m, batch_num_m)]
      score_m = [sum(i) for i in zip(score_m, batch_score_m)]
  
  m_score = [i/j for i,j in zip(score_m, num_m)]
  
  print("Validation Accuracy: ")
  print(m_score)
  v_score = np.mean(m_score)
  print(v_score)
  print()
  # print(s)

  return v_score,m_score,s


def train(model,  optimizer, scheduler, tokenizer, max_epochs, save_path, device, val_freq = 10):
  
  bestpoint_dir = os.path.join(save_path)
  os.makedirs(bestpoint_dir, exist_ok=True)

  global max_accuracy 
  global max_match 
  global val_out 
  # global test_out 
  
  for epoch_i in range(0, max_epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, max_epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):    

        print("batch",step,"out of",len(train_dataloader))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[3].to(device)
        b_token_starts = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_sent_length = batch[4]

        model.zero_grad()   
        model.train()     

        output = model(
          b_input_ids.to(torch.int64),
          b_input_mask.to(torch.int64),
          b_token_starts.to(torch.int64),
          b_sent_length.to(torch.int64),
          b_labels.to(torch.double)
        )
        loss = output[0]

        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        if step % 43 == 0 and step != 0:
          accuracy, match_m, outs = validation(model)

          if(accuracy > max_accuracy):
            # Validation accuracy is the highest
            max_accuracy = accuracy
            max_match = match_m
            val_out = outs
            # test_out = test(model)

            # Saving the test output at best validation accuracy
            os.makedirs(save_path, exist_ok=True)

            val_path = 'val'+str(ind)+'.txt'
            with open(save_path + val_path, "w") as text_file:
              text_file.write(val_out)

            test_path = 'test'+str(ind)+'.txt'
            with open(save_path + test_path, "w") as text_file:
              text_file.write(test_out)

            # To save the model, uncomment the following lines
            torch.save(model.state_dict(), 'mode.pth')
            # model.save_pretrained(bestpoint_dir)  
            print("Saving model bestpoint to ", 'mode.pth')

          print("Best accuracy = "+str(max_accuracy))
          print("")

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
  
  print("")
  print("Training complete!")



model = transformer_model(model_name).to(device)

optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

epochs = num_epochs
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
print(max_accuracy, "\n", max_match)

train(model,  optimizer, scheduler, tokenizer, epochs, save_path, device)