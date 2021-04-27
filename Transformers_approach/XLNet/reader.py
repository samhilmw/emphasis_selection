from config import *
import codecs
from transformers import XLNetConfig, XLNetLMHeadModel, XLNetModel, XLNetTokenizer

# Importing the tokenizer for Transformer model
tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case = False)

def read_token_map(file,word_index = 1,prob_index = 4, caseless = False):
  
  with codecs.open(file, 'r', 'utf-8') as f:
      lines = f.readlines()

  tokenized_texts = []
  token_map = []
  token_labels = []
  sent_length = []

  xlnet_tokens = []
  orig_to_tok_map = []
  labels = []

  xlnet_tokens.append("<s>")
  
  for line in lines:
    if not (line.isspace()):
      feats = line.strip().split()
      word = feats[word_index].lower() if caseless else feats[word_index]
      label = feats[prob_index].lower() if caseless else feats[prob_index]
      labels.append((float)(label))
      orig_to_tok_map.append(len(xlnet_tokens))
      
      if(word == "n't"):
        word = "'t"
        xlnet_tokens[-1] = xlnet_tokens[-1] +"n"

      xlnet_tokens.extend(tokenizer.tokenize(word))
     
    elif len(orig_to_tok_map) > 0:
      xlnet_tokens.append("</s>")
      tokenized_texts.append(xlnet_tokens)
      token_map.append(orig_to_tok_map)
      token_labels.append(labels)
      sent_length.append(len(labels))
      xlnet_tokens = []
      orig_to_tok_map = []
      labels = []
      length = 0
      xlnet_tokens.append("<s>")
          
  if len(orig_to_tok_map) > 0:
    xlnet_tokens.append("</s>")
    tokenized_texts.append(xlnet_tokens)
    token_map.append(orig_to_tok_map)
    token_labels.append(labels)
    sent_length.append(len(labels))
  
  return tokenized_texts, token_map, token_labels, sent_length

def read_test_token_map(file, word_index = 1, caseless = to_case):
  
  with codecs.open(file, 'r', 'utf-8') as f:
      lines = f.readlines()

  tokenized_texts = []
  token_map = []
  sent_length = []

  xlnet_tokens = []
  orig_to_tok_map = []
  
  xlnet_tokens.append("<s>")
  
  for line in lines:
    if not (line.isspace()):
      feats = line.strip().split()
      word = feats[word_index].lower() if caseless else feats[word_index]
      orig_to_tok_map.append(len(xlnet_tokens))
      
      if(word == "n't"):
        word = "'t"
        xlnet_tokens[-1] = xlnet_tokens[-1] +"n"

      xlnet_tokens.extend(tokenizer.tokenize(word))
     
    elif len(orig_to_tok_map) > 0:
      xlnet_tokens.append("</s>")
      tokenized_texts.append(xlnet_tokens)
      token_map.append(orig_to_tok_map)
      sent_length.append(len(orig_to_tok_map))
      xlnet_tokens = []
      orig_to_tok_map = []
      length = 0
      xlnet_tokens.append("<s>")
          
  if len(orig_to_tok_map) > 0:
    xlnet_tokens.append("</s>")
    tokenized_texts.append(xlnet_tokens)
    token_map.append(orig_to_tok_map)
    sent_length.append(len(orig_to_tok_map))
  
  return tokenized_texts, token_map, sent_length


def read_for_output(file, word_index = 1):
  
  with codecs.open(file, 'r', 'utf-8') as f:
      lines = f.readlines()

  words_lsts = []
  word_ids_lsts = []
  words = []
  ids = []
  
  for line in lines:
    if not (line.isspace()):
      feats = line.strip().split()
      words.append(feats[word_index])
      ids.append(feats[0])
     
    elif len(words) > 0:
      words_lsts.append(words)
      word_ids_lsts.append(ids)
      words = []
      ids = []
          
  if len(words) > 0:
    words_lsts.append(words)
    word_ids_lsts.append(ids)
    words = []
    ids = []
  
  return words_lsts , word_ids_lsts
