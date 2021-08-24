import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re 
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 90

class LSTMBaseline(nn.Module):
  def __init__(self,hidden_size,embedding_size,vocabulary_size):
    super().__init__()
    self.hidden_size = hidden_size 

    self.embedding_layer = nn.Embedding(vocabulary_size,embedding_size,padding_idx=0)
    self.lstm_layer = nn.LSTM(embedding_size,hidden_size,batch_first = True,bidirectional=True)        
    self.output_layer_1 = nn.Linear(hidden_size,100)
    self.output_layer_2 = nn.Linear(100,6)

  def forward(self,input_text):
    embeddings = self.embedding_layer(input_text)
    lstm_output,(hidden_state,cell_state) = self.lstm_layer(embeddings)
    linear_output_1 = F.relu(self.output_layer_1(hidden_state[-1,:,:].squeeze(0)))
    linear_output_2 = self.output_layer_2(linear_output_1)
    return linear_output_2

def clean_text(text):
    '''
    Removes punctuation,symbols from the text.
    Input:
        text- String input
    Output:
        text- String
    '''
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]',' ',text) 
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

def string_to_tensor(string_inp,word_to_index_dict):
  string_ids = []
  for char in string_inp:
    if word_to_index_dict.get(char):
      string_ids.append(word_to_index_dict[char])
    else:
      string_ids.append(word_to_index_dict['<UNK>'])
  
  if len(string_ids) > MAX_LEN:
    string_ids = string_ids[:MAX_LEN]
  else:
    string_ids.extend([0]*(MAX_LEN-len(string_ids)))

  string_tensor = torch.tensor(string_ids, device=device, dtype=torch.long)
  return string_tensor

def test_one_sample(text):
  word_to_index_dict = pickle.load(open('./saved_models/char_word_to_index_dict.pkl','rb'))
  class_mapping = pickle.load(open('./saved_models/char_reverse_class_mapping.pkl','rb'))
  char_tensor = string_to_tensor(text,word_to_index_dict) 
  with torch.no_grad():
    model = torch.load('./saved_models/char_lstm_baseline_2.pkl')      
    # model = torch.load('./lstm_baseline_2.pkl')      
    model.eval()
    outputs = model(char_tensor.unsqueeze(0))
    softmax_output = F.softmax(outputs,dim=-1)
    output_preds = torch.argmax(softmax_output,dim=-1)
    return class_mapping[output_preds.item()]