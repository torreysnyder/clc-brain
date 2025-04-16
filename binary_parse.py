import json
import os
from tqdm import tqdm
import nltk
import spacy
import benepar

# Make sure required NLTK packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download the Berkeley Neural Parser model if not present
try:
    benepar.download('benepar_en3')
    print("Using existing benepar_en3 model.")
except:
    print("Downloading benepar_en3 model...")
    benepar.download('benepar_en3')


def load_spacy_benepar():
    """Load spaCy with Berkeley Neural Parser."""
    try:
        nlp = spacy.load('en_core_web_md')
        if 'benepar' not in nlp.pipe_names:
            nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp
    except OSError:
        print("Installing required spaCy model...")
        os.system('python -m spacy download en_core_web_md')
        nlp = spacy.load('en_core_web_md')
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp


def get_binary_parse(doc):
    """Convert a spaCy document to a binary parse string."""
    # Get constituency parse from the first sentence
    parse = list(doc.sents)[0]._.parse_string

    # Convert to binary parse format
    binary_parse = convert_to_binary_parse(parse)
    return binary_parse


def convert_to_binary_parse(parse_string):
    """Convert a Penn Treebank style parse to a binary parse format."""
    # First, parse the string into a tree structure
    from nltk.tree import Tree
    tree = Tree.fromstring(parse_string)

    # Binarize the tree (right-branching)
    binary_tree = binarize_tree(tree)

    # Convert back to string format
    binary_parse = str(binary_tree)

    # Clean up the string to match expected format
    # Remove labels from non-terminals, keep only brackets and terminals
    import re
    binary_parse = re.sub(r'\([A-Z]+\s+', '( ', binary_parse)
    binary_parse = re.sub(r'\([A-Z]+\$?\s+', '( ', binary_parse)
    binary_parse = re.sub(r'\([A-Z]+[A-Z]*-[A-Z]+\s+', '( ', binary_parse)

    return binary_parse


def binarize_tree(tree):
    """Binarize a tree into a right-branching structure."""
    if isinstance(tree, str):
        return tree

    if len(tree) == 1:
        return nltk.Tree(tree.label(), [binarize_tree(tree[0])])

    # For nodes with multiple children, convert to binary branching (right-branching)
    result = nltk.Tree(tree.label(), [binarize_tree(tree[0])])
    current = result

    for child in tree[1:-1]:
        new_node = nltk.Tree(tree.label(), [binarize_tree(child)])
        current = nltk.Tree(tree.label(), [current, new_node])

    if len(tree) > 1:
        current = nltk.Tree(tree.label(), [current, binarize_tree(tree[-1])])

    return current


def process_captions_file(input_file, output_file):
    """Process captions file, parse sentences, and write to a new JSON file."""
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Load spaCy with Berkeley Neural Parser
    nlp = load_spacy_benepar()

    # Process each dictionary in the list
    result = []
    for item in tqdm(data, desc="Processing captions"):
        image_id = item["image_id"]
        captions = item["caption"]

        binary_parses = []
        for caption in captions:
            # Parse the sentence
            doc = nlp(caption)
            binary_parse = get_binary_parse(doc)
            binary_parses.append(binary_parse)

        # Create new dictionary with image_id and binary parses
        result_item = {
            "image_id": image_id,
            "binary_parse": binary_parses
        }
        result.append(result_item)

    # Write to new JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Processed {len(data)} items. Output saved to {output_file}")


if __name__ == "__main__":
    # Define input and output files
    input_file = "test_coco_captions.json"  # Change this to your input file path
    output_file = "test_tree.json"

    # Process the file
    process_captions_file(input_file, output_file)
