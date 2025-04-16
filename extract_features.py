# extract LM features from COCO Captions corpus (for TPDN training)

import os
import codecs
import collections
import argparse
import pandas as pd
from tqdm import tqdm
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModel, AutoTokenizer


#Define classes to hold input examples and features
class InputExample(object):
  def __init__(self, unique_id, text):
    self.unique_id = unique_id
    self.text = text

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

#Load examples from input files
def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  # Read examples from input file and cache unique sentences
  with open(input_file, "r") as json_file:
    caption_ids = json.load(json_file)
  all_captions = pd.DataFrame(caption_ids)
  captioncols = [c for c in all_captions.columns if 'caption' in c]
  for row in tqdm(all_captions.to_dict(orient='records')):
    thesecaptions = [row[c] for c in captioncols]
    item_number = row['image_id']
    for caption in thesecaptions:
      for sub_idx, cap in enumerate(caption):  # Add sub-index to track position within content array
        unique_id = f"{item_number}_{sub_idx}"  # Create a truly unique ID
        examples.append(InputExample(unique_id=unique_id, text=cap))
  return examples





def get_max_seq_length(examples, tokenizer):
  max_seq_len = -1
  for example in examples:
    cand_tokens = tokenizer.tokenize(example.text)
    cur_len = len(cand_tokens)
    if cur_len > max_seq_len:
      max_seq_len = cur_len
  return max_seq_len

# Convert examples to input features for BERT model
def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""
  features = []
  # Convert input examples into features that can be processed by BERT model
  #Args:
  #   examples (list): a list of InputExample objects containing input text
  #   seq_length (int): maximum sequence length to use
  #   tokenizer (BERT tokenizer): convert text to tokens
  #Returns:
  #   list: list of InputFeatures objects, each containing necessary inputs to be encoded by BERT
  for (ex_index, example) in enumerate(examples):
    # tokenize input text
    cand_tokens = tokenizer.tokenize(example.text)
    # Account for [CLS] and [SEP] with "- 2", ensuring that token sequence does not exceed maximum length
    if len(cand_tokens) > seq_length - 2:
      cand_tokens = cand_tokens[0:(seq_length - 2)]

    # Construct input token sequence
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in cand_tokens:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)
    # Convert tokens to input IDs and create input mask, which indicates which tokens are real(1) and which are padding (0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    # Ensure input sizes are correct
    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    #Create an InputFeatures object and add it to list of features
    features.append(
      InputFeatures(
        unique_id=example.unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids))
  return features

# Load BERT model and tokenizer
def load(bert_model):
  print('loading %s model'%bert_model)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(torch.cuda.is_available())
  print(torch.version.cuda)
  # Load pre-trained BERT model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(bert_model)
  model = AutoModel.from_pretrained(bert_model, output_hidden_states=True)
  model.to(device)
  model = torch.nn.DataParallel(model)
  model.eval()
  return model, tokenizer, device

# Save BERT features to output files
def save(model, tokenizer, device):
  # Extract and save BERT features for training and test splits of COCO Captions dataset
  # Args:
    # args: command line arguments containing input/output paths and model settings
    # model: pre-trained BERT model (can be modified for alternative models)
    # tokenizer: BERT tokenizer
    # device: device (CPU/GPU) to run computations on
  #Initialize cache to avoid processing duplicate sentences
  # Process each dataset split and save BERT features
  input_file = "train_coco_captions.json"
  output_file = "bert/train_output_file.json"

  # read examples from input file and update cache
  examples = read_examples(input_file)
  print('%d samples for %s task'%(len(examples), input_file))
    # Convert examples to BERT input features
    # Get max sequence length for this set of examples and add 2 for [CLS] and [SEP] tokens
  examples_map = {example.unique_id: example for example in examples}
  features = convert_examples_to_features(
        examples=examples, seq_length=2+get_max_seq_length(examples,
                                                             tokenizer),
        tokenizer=tokenizer)
    # Convert features to tensors for model input
  all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
  all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    #Create dataset and dataloader for batch processing
  eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler)
    # Initialize progress bar for monitoring
  pbar = tqdm(total=len(examples))
    # Process batches and write features to output file
  with open(output_file, "w", encoding='utf-8') as writer:
    for input_ids, input_mask, example_indices in eval_dataloader:
        # Move inputs to device
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
        #Get BERT embeddings for the batch
        # all_encoder_layers contains hidden states for each leayer
        # pooled_layer contains [CLS] token representation
      outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
      all_encoder_layers = outputs.hidden_states  # all layer hidden states
      pooled_layer = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        #Convert pooled output to numpy
      pooled_layer = pooled_layer.detach().cpu().numpy()
        # Process each example in batch
      for b, example_index in enumerate(example_indices):
          # Get corresponding feature and example
        feature = features[example_index.item()]
        unique_id = feature.unique_id
        item_number = int(unique_id.split('_')[0]) if '_' in str(unique_id) else int(unique_id)
          # Create output JSON
        output_json = collections.OrderedDict()
        output_json["linex_index"] = item_number
        output_json["sentence"] = examples_map[unique_id].text
          # Add pooled output features (CLS token representation)
        output_json["pooled_output"] = [
            round(x.item(), 6) for x in pooled_layer[b]
          ]
          # Process features for each token
        all_out_features = []
        for (i, token) in enumerate(feature.tokens):
            # Collect outputs from all layers for this token
          all_layers = []
          for layer_index in range(len(all_encoder_layers)):
              # Get layer-wise output and convert to numpy
            layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
            layer_output = layer_output[b]
              # Create layer output structure
            layers = collections.OrderedDict()
            layers["index"] = layer_index
            layers["values"] = [
                  round(x.item(), 6) for x in layer_output[i]
              ]
            all_layers.append(layers)
                # Create token feature structure
          out_features = collections.OrderedDict()
          out_features["token"] = token
          out_features["layers"] = all_layers
          all_out_features.append(out_features)
          break # Only process first token
          # Add token features to output
        output_json["features"] = all_out_features
          # Write features to output file
        writer.write(json.dumps(output_json) + "\n")
        # update progress bar
      pbar.update(1)
  pbar.close()
  print('written features to %s'%output_file)


def main():
  # Parse command-line arguments

  # Load BERT model and tokenizer
  bert_model = 'bert-base-uncased'
  model, tokenizer, device = load(bert_model)
  # Save BERT features to output files
  save(model, tokenizer, device)

if __name__ == "__main__":
  main()





