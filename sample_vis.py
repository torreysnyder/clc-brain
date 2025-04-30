import json
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA  # Using PCA instead of UMAP
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter


# Function to create directory if it doesn't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Output directory for plots
output_dir = "embedding_visualizations"
ensure_dir(output_dir)

# Load the model outputs with intermediate activations
print("Loading model outputs with intermediate activations...")
try:
    with open('bert/tpdn_model_bidi_with_activations.json', 'r') as f:
        model_outputs = json.load(f)
    print(f"Successfully loaded {len(model_outputs)} instances")
except FileNotFoundError:
    print("Error: File not found. Please make sure 'bert/tpdn_model_tree_with_activations.json' exists.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON format in the input file.")
    exit(1)

# Limit to first 1000 instances (if there are more)
model_outputs = model_outputs[:1000]
num_instances = len(model_outputs)
print(f"Using first {num_instances} instances for visualization")

# Extract role and filler embeddings from all instances
print("Extracting role and filler embeddings...")
role_embeddings = []
filler_embeddings = []
instance_ids = []
tokens = []
roles = []

for instance in tqdm(model_outputs):
    instance_id = instance["instance_id"]
    instance_tokens = instance["tokens"]
    instance_roles = instance["roles"]

    # Get intermediate activations
    activations = instance["intermediate_activations"]

    # Extract role and filler embeddings
    if "role_embed" in activations and "filler_embed" in activations:
        # Extract embeddings, ensuring we match tokens with their embeddings
        role_embed = np.array(activations["role_embed"]["values"])
        filler_embed = np.array(activations["filler_embed"]["values"])

        # For each token position in the sequence
        seq_length = min(len(instance_tokens), role_embed.shape[1])

        for i in range(seq_length):
            # Store embeddings along with metadata
            role_embeddings.append(role_embed[0, i])
            filler_embeddings.append(filler_embed[0, i])
            instance_ids.append(instance_id)
            tokens.append(instance_tokens[i] if i < len(instance_tokens) else "[PAD]")
            roles.append(instance_roles[i] if i < len(instance_roles) else "[PAD]")

# Convert lists to numpy arrays
role_embeddings = np.array(role_embeddings)
filler_embeddings = np.array(filler_embeddings)

print(f"Collected {len(role_embeddings)} role embeddings and {len(filler_embeddings)} filler embeddings")
print(f"Role embedding shape: {role_embeddings.shape}, Filler embedding shape: {filler_embeddings.shape}")

# Find the 20 most frequent roles
role_counter = Counter(roles)
top_20_roles = [role for role, count in role_counter.most_common(20)]
print(f"Top 20 most frequent roles: {top_20_roles}")

# Add a verification step to check if roles with the same label have identical embeddings
print("Verifying if identical role labels have identical embeddings...")
for role in top_20_roles[:5]:  # Check first 5 of the top 20 roles
    role_indices = [i for i, r in enumerate(roles) if r == role]
    if len(role_indices) > 1:
        # Get embeddings for this role
        role_vecs = role_embeddings[role_indices]
        # Check if all vectors are identical
        reference_vec = role_vecs[0]
        all_identical = all(np.array_equal(vec, reference_vec) for vec in role_vecs)
        # Calculate maximum difference
        max_diff = np.max(np.abs(role_vecs - reference_vec)) if len(role_vecs) > 1 else 0
        print(f"Role {role}: All vectors identical? {all_identical}, Max difference: {max_diff}")

# Filter embeddings to only include top 20 roles
filtered_indices = [i for i, role in enumerate(roles) if role in top_20_roles]
filtered_role_embeddings = role_embeddings[filtered_indices]
filtered_roles = [roles[i] for i in filtered_indices]
filtered_instance_ids = [instance_ids[i] for i in filtered_indices]

print(f"Filtered to {len(filtered_role_embeddings)} embeddings for top 20 most frequent roles")

# Apply PCA to reduce dimensionality to 2D for filtered role embeddings
print("Applying PCA to filtered role embeddings...")
role_pca = PCA(n_components=2, random_state=100)
filtered_role_embedding_2d = role_pca.fit_transform(filtered_role_embeddings)
print(f"Role PCA explained variance: {role_pca.explained_variance_ratio_}")

# Apply PCA to filler embeddings as in the original code
print("Applying PCA to filler embeddings...")
filler_pca = PCA(n_components=2, random_state=42)
filler_embedding_2d = filler_pca.fit_transform(filler_embeddings)
print(f"Filler PCA explained variance: {filler_pca.explained_variance_ratio_}")

# Create a color map for the top 20 roles
role_color_map = {role: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                  for i, role in enumerate(top_20_roles)}

# Define special tokens
special_tokens = ['[CLS]', '[SEP]', '[UNK]', '[PAD]']


# Function to visualize embeddings with plotly and hover annotations
def visualize_embeddings_html(embedding_2d, hover_text, color_values, title, filename,
                              color_labels=None, color_discrete_map=None, instance_ids=None):
    # Create a plotly figure
    if color_discrete_map:
        fig = px.scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            color=color_values,
            hover_name=hover_text,
            opacity=0.7,
            title=title,
            color_discrete_map=color_discrete_map,
            labels={'color': 'Category'}
        )
    else:
        fig = px.scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            color=color_values,
            hover_name=hover_text,
            opacity=0.7,
            title=title,
            labels={'color': 'Category'}
        )

    # Add hover data with additional information
    for point_idx in range(len(hover_text)):
        instance_id = instance_ids[point_idx] if instance_ids else "N/A"
        id_display = instance_id.split('_')[-1] if isinstance(instance_id, str) else instance_id

        fig.add_trace(
            go.Scatter(
                x=[embedding_2d[point_idx, 0]],
                y=[embedding_2d[point_idx, 1]],
                mode='markers',
                marker=dict(opacity=0),
                hoverinfo='text',
                hovertext=f"Text: {hover_text[point_idx]}<br>Category: {color_values[point_idx]}<br>Instance: {id_display}",
                showlegend=False
            )
        )

    # Improve layout
    fig.update_layout(
        legend_title_text='Categories',
        width=1000,
        height=800,
        template='plotly_white',
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # Save as interactive HTML
    fig.write_html(os.path.join(output_dir, filename))
    print(f"Saved interactive visualization to {os.path.join(output_dir, filename)}")


# Visualize the filtered role embeddings (top 20 most frequent roles)
print("Generating interactive role embeddings visualization (top 20 most frequent roles)...")
visualize_embeddings_html(
    filtered_role_embedding_2d,
    filtered_roles,  # Hover text shows roles
    filtered_roles,  # Color by role
    "PCA Visualization of Bidi Role Embeddings (Top 20 Most Frequent Roles)",
    "role_embedding_pca_bidi_top20.html",
    color_discrete_map=role_color_map,
    instance_ids=filtered_instance_ids
)

# Define color map for token categories
token_color_map = {token: px.colors.qualitative.Set1[i] for i, token in enumerate(special_tokens)}
token_color_map["Regular Token"] = "lightgray"

# For tokens, we'll categorize them
token_categories = []
for token in tokens:
    if token in special_tokens:
        token_categories.append(token)  # Special tokens get their own category
    else:
        token_categories.append("Regular Token")  # All other tokens grouped together

# Visualize filler embeddings with hover text showing tokens (unchanged from original)
print("Generating interactive filler embeddings visualization...")
visualize_embeddings_html(
    filler_embedding_2d,
    tokens,  # Hover text shows tokens
    token_categories,  # Color by token category (special vs regular)
    "PCA Visualization of Bidi Filler Embeddings",
    "filler_embedding_pca_bidi.html",
    color_discrete_map=token_color_map,
    instance_ids=instance_ids
)

# Generate additional visualization with token frequency information
print("Generating token frequency visualization...")

# Count token frequencies
token_counts = {}
for t in tokens:
    if t not in token_counts:
        token_counts[t] = 0
    token_counts[t] += 1

# Find top 20 most frequent tokens (excluding special tokens)
freq_tokens = sorted([(t, c) for t, c in token_counts.items() if t not in special_tokens],
                     key=lambda x: x[1], reverse=True)[:20]

# Create frequency-based categories
token_freq_categories = []
for token in tokens:
    if token in special_tokens:
        token_freq_categories.append(token)
    elif token in [t for t, _ in freq_tokens]:
        token_freq_categories.append(f"Frequent: {token}")
    else:
        token_freq_categories.append("Other Token")

# Create combined hover text with token and frequency
hover_texts = []
for token in tokens:
    freq = token_counts.get(token, 0)
    hover_texts.append(f"{token} (Count: {freq})")

# Define color map for frequent tokens
freq_color_map = {token: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                  for i, token in enumerate(special_tokens)}
for i, (token, _) in enumerate(freq_tokens):
    freq_color_map[f"Frequent: {token}"] = px.colors.sequential.Plasma[
        int(i * len(px.colors.sequential.Plasma) / len(freq_tokens))]
freq_color_map["Other Token"] = "lightgray"

# Create the frequency-based visualization
print("Generating interactive token frequency visualization...")
visualize_embeddings_html(
    filler_embedding_2d,
    hover_texts,  # Hover text with token and frequency info
    token_freq_categories,  # Color by frequency category
    "PCA Visualization of Bidi Filler Embeddings - Frequency",
    "filler_embeddings_pca_bidi_frequency.html",
    instance_ids=instance_ids
)

print("All interactive visualizations complete!")
