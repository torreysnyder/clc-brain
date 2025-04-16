# Tensor Product Decomposition Network to approximate BERT

import sys
import json
import pandas as pd
import io
import os
from tqdm import tqdm
import argparse
import random
import numpy as np
import codecs

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer
from treerole_helper import gen_treerole, gen_rand_tree


def set_seed(seed):
  use_cuda = torch.cuda.is_available()
  print(use_cuda)
  random.seed(seed)
  torch.manual_seed(seed)
  if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
  device = torch.device("cuda" if use_cuda else "cpu")
  return device


def load_bert(bert_model):
  # load bert tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained(bert_model,
                                            do_lower_case=True)
  pretrained_model = AutoModel.from_pretrained(bert_model, output_hidden_states=True)
  return tokenizer, pretrained_model


# role scheme generator
def get_roles(role_scheme, rand_tree, tokens, parse_info=None, unmasked_tokens=None):
  if role_scheme == 'l2r':
    return [i for i in range(len(tokens))]
  if role_scheme == 'r2l':
    return [len(tokens) - i - 1 for i in range(len(tokens))]
  if role_scheme == 'bow':
    return [0] * len(tokens)
  if role_scheme == 'bidi':
    return ['%d-%d' % (i, len(tokens) - i - 1) for i in range(len(tokens))]
  if role_scheme == 'tree':
    tokens = unmasked_tokens if unmasked_tokens else tokens
    tree_roles = None
    if not rand_tree:
      tree_roles = gen_treerole(tokens[1:-1], parse_info)
    else:
      tree_roles = gen_rand_tree(len(tokens) - 2)
    return ['[CLS]'] + tree_roles + ['[SEP]']


# create word dictionaries
def create_vocab(tokenizer):
  id2token = {0: '[UNK]', 1: '[CLS]', 2: '[SEP]'}
  token2id = {'[UNK]': 0, '[CLS]': 1, '[SEP]': 2}

  # Check if input file exists
  if not os.path.exists("bert/train_output_file.json"):
    print("ERROR: File 'bert/train_output_file.json' does not exist.")
    sys.exit(1)

  with open("bert/train_output_file.json", 'r') as f:
    for line in f:
      try:
        content = json.loads(line.strip())
        premise = content.get('sentence', '')

        # Tokenize and print
        tokens = ['[CLS]'] + tokenizer.tokenize(premise) + ['[SEP]']
        #print(f"Premise: {premise}")
        #print(f"Tokens: {tokens}")

        # Add tokens to vocabulary
        for token in tokens:
          if token not in token2id:
            token2id[token] = len(id2token)
            id2token[len(id2token)] = token

      except Exception as e:
        print(f"Error processing line: {e}")

  print('Vocabulary size = %d' % len(id2token))
  return id2token, token2id


# load pretrained embeddings
def load_pretrained_embed(pretrained, tokenizer, id2token):
  weights = pretrained.embeddings.word_embeddings.weight.detach().numpy()
  pretrained_embeddings = np.random.rand(len(id2token), weights.shape[1])
  bert_token_ids = tokenizer.convert_tokens_to_ids([id2token[ti] for ti in range(len(id2token))])
  assert (len(bert_token_ids) == len(id2token))
  for i, bti in enumerate(bert_token_ids):
    pretrained_embeddings[i] = weights[bti]
  print('loaded pretrained embeddings')
  return pretrained_embeddings


# create model inputs
class Instance(object):
  def __init__(self, filler, role, rep):
    self.filler = filler
    self.role = role
    self.rep = rep


def get_representation(content, layer):
  if layer == -1:
    return content['pooled_output']
  return content['features'][0]['layers'][layer]['values']


id2role, role2id = {0: '[UNK]'}, {'[UNK]': 0}


def read_as_tensors(inp_file, tokenizer, token2id, device):
  """
  Read input data from a JSON file and convert to tensors for model processing.
  Also logs problematic tokens to a text file and tracks their count.

  Args:
      inp_file (str): Path to the input JSON file
      tokenizer (AutoTokenizer): BERT tokenizer for processing text
      token2id (dict): Dictionary mapping tokens to their unique IDs
      device (torch.device): Device to move tensors to (CPU/GPU)

  Returns:
      list: A list of Instance objects containing processed tensors
  """
  # Initialize counters and data list
  data = []
  total_problematic_tokens = 0
  unique_problematic_tokens = set()


  # Check if input file exists
  if not os.path.exists(inp_file):
    print(f"ERROR: Input file {inp_file} does not exist.")
    sys.exit(1)

  # Create/open a file to log problematic tokens and missing parses
  log_filename = 'l2r_problematic_tokens.txt'
  with open(log_filename, 'w') as log_file:
    log_file.write("Processing Log\n\n")
    log_file.write("PROBLEMATIC TOKENS:\n\n")

    # Open and iterate through the input file
    with open(inp_file, 'r') as f:
      lines = f.readlines()
      total_lines = len(lines)
      print(f"Processing {total_lines} lines from {inp_file}")

      for line_num, line in enumerate(lines, 1):
        try:
          # Parse each line as a JSON object
          content = json.loads(line.strip())

          # Extract the sentence from the content, defaulting to empty string
          premise = content.get('sentence', '')
          if not premise:
            print(f"WARNING: Line {line_num} has no sentence. Skipping.")
            continue

          # Tokenize the sentence with special BERT tokens
          tokens = ['[CLS]'] + tokenizer.tokenize(premise) + ['[SEP]']

          # Convert tokens to their corresponding IDs
          filler_ids = []
          problematic_tokens = []
          for token in tokens:
            if token in token2id:
              filler_ids.append(token2id[token])
            else:
              problematic_tokens.append(token)
              unique_problematic_tokens.add(token)
              filler_ids.append(token2id['[UNK]'])

          # Update total count and log problematic tokens
          total_problematic_tokens += len(problematic_tokens)

          if problematic_tokens:
            log_entry = f"Line {line_num}:\n"
            log_entry += f"Original sentence: {premise}\n"
            log_entry += f"Unknown tokens: {problematic_tokens}\n"
            log_entry += f"Full token sequence: {tokens}\n"
            log_entry += "-" * 80 + "\n\n"
            log_file.write(log_entry)
            #print(f"Line {line_num}: Unknown tokens {problematic_tokens}")

          # Create filler tensor
          filler_t = Variable(torch.LongTensor(filler_ids), requires_grad=False).unsqueeze(0)
          filler_t = filler_t.to(device)

          # Get the appropriate binary parse information


          # Generate roles for the tokens
          roles = get_roles('bow', True, tokens,
                            parse_info=content.get('binaryParse'),
                            unmasked_tokens=None)

          # Convert roles to their corresponding IDs
          role_ids = []
          for role in roles:
            if role not in role2id:
              role2id[role] = len(id2role)
              id2role[role2id[role]] = role
            role_ids.append(role2id[role])

          # Create role tensor
          role_t = Variable(torch.LongTensor(role_ids), requires_grad=False).unsqueeze(0)
          role_t = role_t.to(device)

          # Extract and create representation tensor
          representation = get_representation(content, -1)
          rep_t = Variable(torch.FloatTensor(representation), requires_grad=False).unsqueeze(0)
          rep_t = rep_t.to(device)

          # Create Instance object
          data.append(Instance(filler_t, role_t, rep_t))

          # Print progress periodically
          if line_num % 100 == 0 or line_num == total_lines:
            print(f"Processed {line_num}/{total_lines} lines ({line_num / total_lines * 100:.1f}%)")

        except Exception as e:
          error_msg = f"WARNING processing line {line_num}: {e}\n"
          error_msg += f"Problematic line content: {line.strip()}\n"
          error_msg += "-" * 80 + "\n\n"
          log_file.write(error_msg)
          #print(f"WARNING processing line {line_num}: {e}")
          # Continue processing other lines instead of exiting

    # Write summary statistics to the log file
    summary = "\nSUMMARY:\n"
    summary += f"Total lines processed: {total_lines}\n"
    summary += f"Successful instances created: {len(data)}\n"
    summary += f"Total problematic tokens encountered: {total_problematic_tokens}\n"
    summary += f"Number of unique problematic tokens: {len(unique_problematic_tokens)}\n"
    summary += f"Unique problematic tokens: {sorted(list(unique_problematic_tokens))}\n"
    log_file.write(summary)

  print(f'Read {len(data)} instances from {inp_file}')
  print(f'Total problematic tokens encountered: {total_problematic_tokens}')
  print(f'Number of unique problematic tokens: {len(unique_problematic_tokens)}')
  print(f'Processing details logged to {log_filename}')

  if len(data) == 0:
    print(f"ERROR: No data instances were successfully processed from {inp_file}")
    sys.exit(1)

  return data


def read_tensors(tokenizer, token2id, device):
  print("Loading training data from bert/train_output_file.json...")
  train_data = read_as_tensors("bert/train_output_file.json", tokenizer, token2id, device)
  print(f"Loaded {len(train_data)} training instances")

  print("Loading test data from bert/test_subordination_file.json...")
  test_data = read_as_tensors("bert/test_output_file.json", tokenizer, token2id, device)
  print(f"Loaded {len(test_data)} test instances")

  print('role count = %d' % len(id2role))

  if not train_data:
    print("ERROR: No training data was loaded.")
    sys.exit(1)

  if not test_data:
    print("ERROR: No test data was loaded.")
    sys.exit(1)

  return train_data, test_data


# Model definition
# Defines the tensor product, used in tensor product representations
class SumFlattenedOuterProduct(nn.Module):
  def __init__(self):
    super(SumFlattenedOuterProduct, self).__init__()

  def forward(self, input1, input2):
    # This layer will take the sum flattened outer product of the filler
    # and role embeddings
    # outer_product = torch.mm(input1.t(), input2)
    # flattened_outer_product = outer_product.view(-1).unsqueeze(0)
    outer_product = torch.bmm(input1.transpose(1, 2), input2)
    flattened_outer_product = outer_product.view(outer_product.size()[0], -1).unsqueeze(0)
    sum_flattened_outer_product = flattened_outer_product
    return sum_flattened_outer_product


# A tensor product encoder layer
# Takes a list of fillers and a list of roles and returns an encoding
class TensorProductEncoder(nn.Module):
    def __init__(self, num_roles, num_fillers, role_dim, filler_dim,
                 final_layer_width, pretrained_embeddings, untrained):
      super(TensorProductEncoder, self).__init__()
      self.role_embed = nn.Embedding(num_roles, role_dim)
      self.filler_embed = None
      if pretrained_embeddings is None:
        self.filler_embed = nn.Embedding(num_fillers, filler_dim)
      else:
        filler_dim = pretrained_embeddings.shape[1]
        pretrained_embed = nn.Embedding(num_fillers, filler_dim)
        if not untrained:
          pretrained_embed.weight.data = torch.from_numpy(pretrained_embeddings).float()
        pretrained_embed.weight.requires_grad = False
        self.filler_embed = nn.Sequential(pretrained_embed,
                                          nn.Linear(filler_dim, filler_dim)) if not untrained else pretrained_embed
      if untrained:
        self.filler_embed.weight.requires_grad = False
        self.role_embed.weight.requires_grad = False
      self.sum_layer = SumFlattenedOuterProduct()
      self.last_layer = nn.Linear(filler_dim * role_dim, final_layer_width)

      # Hook for capturing intermediate activations
      #self.intermediate_activations = {}

      # Register hooks for capturing activations
      #self.role_embed.register_forward_hook(self._capture_role_embed_hook)
      #self.filler_embed.register_forward_hook(self._capture_filler_embed_hook)
      #self.sum_layer.register_forward_hook(self._capture_sum_layer_hook)
      #self.last_layer.register_forward_hook(self._capture_last_layer_hook)

    #def _capture_role_embed_hook(self, module, input, output):
      #self.intermediate_activations['role_embed'] = output.detach()

    #def _capture_filler_embed_hook(self, module, input, output):
      #self.intermediate_activations['filler_embed'] = output.detach()

    #def _capture_sum_layer_hook(self, module, input, output):
      #self.intermediate_activations['sum_layer'] = output.detach()

    #def _capture_last_layer_hook(self, module, input, output):
      #self.intermediate_activations['last_layer'] = output.detach()

    def forward(self, fillers, roles):
      # Clear previous activations
      #self.intermediate_activations.clear()

      filler_embed = self.filler_embed(fillers)
      role_embed = self.role_embed(roles)
      output = self.sum_layer(filler_embed, role_embed)
      output = self.last_layer(output)
      return output


# batchify the data
def batchify(data, batch_size):
  len2inst = {}
  for di, datum in enumerate(data):
    dlen = datum.filler.size(1)
    if dlen not in len2inst:
      len2inst[dlen] = []
    len2inst[dlen].append(di)
  batches = []
  for dlen in sorted(len2inst.keys()):
    ids = len2inst[dlen]
    for bi in range(len(ids)//batch_size):
      first_item = data[ids[bi*batch_size]]
      cur_filler, cur_role, cur_rep = first_item.filler, first_item.role, \
                                      first_item.rep
      for idi in ids[(bi*batch_size)+1: (bi+1)*batch_size]:
        inst = data[idi]
        cur_filler = torch.cat((cur_filler, inst.filler), 0)
        cur_role = torch.cat((cur_role, inst.role), 0)
        cur_rep = torch.cat((cur_rep, inst.rep), 0)
      batches.append((cur_filler, cur_role, cur_rep.unsqueeze(0)))
  return batches

def run_tpdn(role_dim, filler_dim, untrained, id2token, token2id, pretrained_embeddings, train_data, test_data,
               device):
    """
    Train and evaluate the Tensor Product Decomposition Network with intermediate activation saving.
    """
    if not train_data:
      print("Error: No training data was loaded. Please check your input files and tree parse information.")
      return None
    rep_size = train_data[0].rep.size(1)

    # Create the model
    print(f"Creating model with {len(role2id)} roles, {len(token2id)} tokens, {role_dim} role dimensions, "
          f"{filler_dim} filler dimensions, and {rep_size} representation size")
    model = TensorProductEncoder(len(role2id), len(token2id), role_dim,
                                 filler_dim, train_data[0].rep.size(1), pretrained_embeddings,
                                 untrained)
    model.to(device)

    # Training configuration (remains the same as before)
    batch_size = 50
    learning_rate = 0.0001
    n_epochs = 100
    patience = 10
    best_loss = float('inf')
    training_loss = []
    test_loss = []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # (Rest of the existing training loop code remains the same)
    def calculate_mse(data):
      """Calculate MSE consistently for both train and test data"""
      total_mse = 0.
      for instance in data:
        model_output = model(instance.filler, instance.role).data
        instance_mse = torch.mean(torch.pow(model_output - instance.rep.data, 2))
        total_mse += instance_mse
      return total_mse / len(data)

    def train_batch(batch):
      """Train on a batch while tracking true MSE"""
      optimizer.zero_grad()
      total_loss = 0

      for items in batch:
        filler_in, role_in, target_out = items
        enc_out = model(filler_in, role_in)
        loss = torch.mean(torch.pow(enc_out - target_out, 2))
        total_loss += loss

      # Backward pass and optimization
      total_loss.backward()
      optimizer.step()
      return total_loss.item() / len(batch)

    # Batchify train data
    train_batches = batchify(train_data, batch_size)

    # Training loop
    for epoch in range(n_epochs):
      # Training phase
      model.train()
      random.shuffle(train_batches)

      # Process batches
      for bi in tqdm(range(len(train_batches) // batch_size)):
        _ = train_batch(train_batches[bi * batch_size: (bi + 1) * batch_size])

      # Evaluation phase
      model.eval()
      with torch.no_grad():
        # Calculate losses consistently using the same method
        train_epoch_loss = calculate_mse(train_data)
        test_epoch_loss = calculate_mse(test_data)

      # Track losses
      training_loss.append(train_epoch_loss)
      test_loss.append(test_epoch_loss)

      print(f'Epoch {epoch + 1}: Train Loss = {train_epoch_loss:.4f}, Test Loss = {test_epoch_loss:.4f}')

      # Early stopping
      if test_epoch_loss < best_loss:
        best_loss = test_epoch_loss
        torch.save(model.state_dict(), 'bert/tpdn_output_cocoIDs_bow.json')
        patience = 10
      else:
        patience -= 1
        if patience == 0:
          print("Early stopping triggered")
          break

    # Plot loss progression
    plt.figure(figsize=(10, 5))
    plt.plot([loss.cpu().numpy() for loss in training_loss], label='Training Loss')
    plt.plot([loss.cpu().numpy() for loss in test_loss], label='Test Loss')
    plt.title('LR = 0.0001 Bow Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('LR_0.0001_batch_50_bow_loss_progression.png')
    plt.close()

    # First, load the original test_output_file.json to extract the linex_index values
    original_linex_indices = {}
    try:
      with open("bert/test_output_file.json", 'r') as f:
        for i, line in enumerate(f):
        # Only need the first 1000 entries
          content = json.loads(line.strip())
          # Use the index in the file as the key, map to the linex_index from the content
          original_linex_indices[i] = content.get("linex_index", i)
    except Exception as e:
      print(f"Warning: Could not load linex_index values from test_output_file.json: {e}")
      print("Will use sequential instance IDs instead.")
    # After training, we'll enhance the final evaluation to save intermediate activations
    model.load_state_dict(torch.load('bert/tpdn_output_cocoIDs_bow.json'))
    print("Generating model outputs with intermediate activations for first 1000 test data instances...")
    model_outputs_with_activations = []
    with torch.no_grad():
      # Only process the first 10 instances from test_data
      for i, instance in enumerate(test_data):
        # Perform forward pass
        output = model(instance.filler, instance.role).squeeze(0)
        target = instance.rep.squeeze(0)

        # Collect intermediate activations
        #intermediate_activations = model.intermediate_activations

        # Prepare instance details
        output_data = output.cpu().numpy().tolist()
        target_data = target.cpu().numpy().tolist()
        filler_ids = instance.filler.squeeze(0).cpu().numpy().tolist()
        role_ids = instance.role.squeeze(0).cpu().numpy().tolist()
        tokens = [id2token.get(fid, '[UNK]') for fid in filler_ids]
        roles = [id2role.get(rid, '[UNK]') for rid in role_ids]
        instance_loss = torch.mean(torch.pow(output - target, 2)).item()

        # Prepare intermediate activations for saving
        #activations_to_save = {}
        #for key, activation in intermediate_activations.items():
          #activations_to_save[key] = {
            #'shape': list(activation.shape),
            #'values': activation.cpu().numpy().tolist()
          #}
        instance_id = original_linex_indices.get(i, i)
        # Create entry with full details
        entry = {
          "instance_id": instance_id,
          "tokens": tokens,
          "roles": roles,
          "model_output": output_data,
          "target_output": target_data,
          "instance_loss": float(instance_loss),
          #"intermediate_activations": activations_to_save
        }
        model_outputs_with_activations.append(entry)

    # Save model outputs with intermediate activations
    output_path = 'bert/tpdn_model_bow_with_cocoIDs.json'
    with open(output_path, 'w') as f:
      json.dump(model_outputs_with_activations, f, indent=2)
    print(f"Model outputs with intermediate activations for first 10 instances saved to {output_path}")

    return model
def main():
  device = set_seed(100)
  print(device)
  bert_model = 'bert-base-uncased'
  tokenizer, pretrained_model = load_bert(bert_model)
  id2token, token2id = create_vocab(tokenizer)
  pretrained_embeddings = load_pretrained_embed(pretrained_model, tokenizer, id2token)
  train_data, test_data = read_tensors(tokenizer, token2id, device)
  if not train_data or not test_data:
    print("ERROR: No data loaded. Please check the input files and tree parse information.")
    return
  run_tpdn(6, 10, True, id2token, token2id, pretrained_embeddings, train_data, test_data, device)

if __name__ == "__main__":
  main()

