# %%
from collections import defaultdict
from typing import Tuple, Union, List, Optional, Dict, Callable
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor, device as TorchDevice
import torch as t
from torch.utils.data import DataLoader, TensorDataset

import plotly.express as px
import pandas as pd
import transformer_lens as tl
from einops import einsum
from word2word import Word2word

from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import (
    get_most_similar_embeddings,
    remove_hooks,
    repo_path_to_abs_path,
)

# %% ----------------------- model setup ------------------------
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out

# %% ---------------------- functions ----------------------
def generate_tokens(model: tl.HookedTransformer, n_toks: int, device: TorchDevice) -> Tuple[Tensor, Tensor]:
    """
    Processes tokens to find corresponding tokens in another language (English to French) and returns their indices.

    Args:
        model: The transformer model used for token processing.
        n_toks: The number of tokens to process.
        device: The device on which tensors will be allocated.

    Returns:
        A tuple of tensors containing indices of English and French tokens.
    """
    en2fr = Word2word("en", "fr")
    en_toks, fr_toks = [], []
    en_strs, fr_strs = [], []
    for tok in range(n_toks):
        en_tok_str = model.to_string([tok])
        assert type(en_tok_str) == str
        if len(en_tok_str) < 7 or en_tok_str[0] != " ":
            continue
        try:
            fr_tok_str = " " + en2fr(en_tok_str[1:], n_best=1)[0]
        except Exception:
            continue
        if en_tok_str.lower() == fr_tok_str.lower():
            continue
        try:
            fr_tok = model.to_single_token(fr_tok_str)
        except Exception:
            continue
        en_toks.append(tok)
        fr_toks.append(fr_tok)
        en_strs.append(en_tok_str)
        fr_strs.append(fr_tok_str)
        print("en: ", en_tok_str)
        print("fr: ", fr_tok_str)
    return t.tensor(en_toks, device=device), t.tensor(fr_toks, device=device)

def generate_google_words(model: tl.HookedTransformer, n_toks: int, device: TorchDevice, en_file: List[str]) -> Tuple[List[str], List[str]]:
    """
    Takes in a list of English strings then returns an English and French list. The French list
    contains French translations of English words for which a translation can be found.

    Args:
        model: The transformer model used for token processing.
        n_toks: The number of tokens to process.
        device: The device on which tensors will be allocated.
        en_file: List of English strings to be translated.

    Returns:
        A tuple of lists containing English strings and their French translations.
    """
    en2fr = Word2word("en", "fr")
    en_toks_list, fr_toks_list = [], []
    en_strs_list, fr_strs_list = [], []
    for i in range(n_toks):
        try:
            en_str = en_file[i]
            en_toks = model.to_tokens(en_str)
            fr_str = en2fr(en_str, n_best=1)[0]
            fr_toks = model.to_tokens(fr_str)
        except Exception:
            continue
        print(en_str)
        print(fr_str)
        en_toks_list.append(en_toks)
        fr_toks_list.append(fr_toks)
        en_strs_list.append(en_str)
        fr_strs_list.append(fr_str)
    return en_strs_list, fr_strs_list

def generate_embeddings(model: tl.HookedTransformer, en_toks: Tensor, fr_toks: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Generates embeddings for English and French tokens.

    Args:
        model: The transformer model used for generating embeddings.
        en_toks: Tensor of English token indices.
        fr_toks: Tensor of French token indices.

    Returns:
        A tuple of tensors containing embeddings for English and French tokens.
    """
    en_embeds = model.embed.W_E[en_toks].detach().clone()
    fr_embeds = model.embed.W_E[fr_toks].detach().clone()
    return en_embeds, fr_embeds

def create_data_loaders(en_embeds: Tensor, fr_embeds: Tensor, batch_size: int, train_ratio: float = 1.0, en_attn_mask: Optional[Tensor] = None, fr_attn_mask: Optional[Tensor] = None, match_dims: bool = False, mask: bool = False) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Refactored function to create data loaders for training and optionally testing datasets from embedding tensors and attention masks, with an option to match dimensions and apply masks.

    Args:
        en_embeds: Tensor of English embeddings.
        fr_embeds: Tensor of French embeddings.
        batch_size: The size of each batch.
        train_ratio: The ratio of the dataset to be used for training.
        en_attn_mask: Optional attention mask for English embeddings.
        fr_attn_mask: Optional attention mask for French embeddings.
        match_dims: Whether to match the dimensions of English and French embeddings.
        mask: Whether to apply the attention masks to the embeddings.

    Returns:
        A DataLoader for the training dataset, and optionally a DataLoader for the testing dataset.
    """
    # Match dimensions if required
    if match_dims:
        min_len = min(len(en_embeds), len(fr_embeds))
        en_embeds, fr_embeds = en_embeds[:min_len], fr_embeds[:min_len]
        if mask and en_attn_mask is not None and fr_attn_mask is not None:
            min_len_mask = min(len(en_attn_mask), len(fr_attn_mask))
            en_attn_mask, fr_attn_mask = en_attn_mask[:min_len_mask], fr_attn_mask[:min_len_mask]

    # Apply masks if required
    if mask:
        if en_attn_mask is not None:
            en_embeds = en_embeds * en_attn_mask
        if fr_attn_mask is not None:
            fr_embeds = fr_embeds * fr_attn_mask

    # Create dataset based on available data
    dataset_tensors = [en_embeds, fr_embeds]
    if en_attn_mask is not None and fr_attn_mask is not None:
        dataset_tensors.extend([en_attn_mask, fr_attn_mask])
    dataset = t.utils.data.TensorDataset(*dataset_tensors)

    # Split dataset into training and testing if train_ratio is specified
    if train_ratio != 1.0:
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size
        print(train_size, test_size)
        train_set, test_set = t.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = t.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = t.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader
    else:
        train_loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader

def initialize_transform_and_optim(d_model: int, device: TorchDevice, transformation: str, lr: float) -> Tuple[Union[Module, Tensor], Optimizer]:
    """
    Initializes a transformation and its corresponding optimizer based on the specified transformation type.

    Args:
        d_model: The dimensionality of the model embeddings.
        device: The device on which the transformation will be allocated.
        transformation: The type of transformation to initialize.
        lr: Learning rate for the optimizer.

    Returns:
        A tuple containing the transformation and its optimizer.
    """
    if transformation == "translate0":
        transform = t.zeros([d_model], device=device, requires_grad=True)
        optim = t.optim.Adam([transform], lr=0.0002)
    elif transformation == "translate_2":
        transform = t.zeros([d_model], device=device, requires_grad=True)
        optim = t.optim.Adam([transform], lr=0.0002)
    elif transformation == "rotation":
        transform = t.nn.Linear(d_model, d_model, bias=False, device=device)
        # optim = t.optim.Adam(list(learned_rotation.parameters()) + [translate], lr=0.0002)
        optim = t.optim.Adam(transform.parameters(), lr=lr)
    elif transformation == "linear_map":
        initial_rotation = t.nn.Linear(d_model, d_model, bias=False, device=device)
        transform = t.nn.utils.parametrizations.orthogonal(initial_rotation, "weight")
        optim = t.optim.Adam(list(transform.parameters()), lr=0.0002)
        # optim = t.optim.Adam(list(linear_map.parameters()) + [translate], lr=0.01)
    else:
        raise Exception("the supplied transform was unrecognized")
    return transform, optim

def word_pred_from_embeds(embeds: Tensor, rotation: Union[Module, Tensor], lerp: float = 1.0) -> Tensor:
    """
    Applies a rotation transformation to the input embeddings.

    Args:
        embeds (Tensor): The input embeddings to be transformed.
        rotation (Union[Module, Tensor]): The rotation transformation to be applied. Can be a PyTorch module or a tensor.
        lerp (float, optional): Linear interpolation factor. Defaults to 1.0, meaning full rotation is applied.

    Returns:
        Tensor: The transformed embeddings after applying the rotation.
    """
    # Assuming `learned_rotation` is a function or module that applies the rotation transformation.
    return rotation(embeds)

def word_distance_metric(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """
    Computes the negative cosine similarity between two tensors.

    Args:
        a (t.Tensor): The first tensor.
        b (t.Tensor): The second tensor.

    Returns:
        t.Tensor: The negative cosine similarity between the input tensors.
    """
    return -t.nn.functional.cosine_similarity(a, b)

def train_and_evaluate(model: tl.HookedTransformer, device: TorchDevice, train_loader: DataLoader, test_loader: DataLoader, initial_rotation: Union[Module, Tensor], optim: Optimizer, n_epochs: int) -> Union[Module, Tensor]:
    """
    Trains and evaluates the model, returning the learned transformation.

    Args:
        model: The transformer model used for training.
        device: The device on which the model is allocated.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
        initial_rotation: The initial transformation to be optimized.
        optim: The optimizer for the transformation.
        n_epochs: The number of epochs to train for.

    Returns:
        The learned transformation after training.
    """
    loss_history = []
    initial_rotation.to(device)  # Ensure the learned_rotation model is on the correct device
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch_idx, (en_embed, fr_embed) in enumerate(train_loader):
            print(batch_idx)
            en_embed =  en_embed.to(device)
            fr_embed = fr_embed.to(device)

            optim.zero_grad()
            pred = word_pred_from_embeds(en_embed, initial_rotation)
            loss = word_distance_metric(pred, fr_embed).mean()
            loss_history.append(loss.item())
            loss.backward()
            optim.step()
            epoch_pbar.set_description(f"Loss: {loss.item():.3f}")
    px.line(y=loss_history, title="Loss History").show()
    learned_rotation = initial_rotation
    evaluate_accuracy(model, device, test_loader, learned_rotation)
    return learned_rotation

def evaluate_accuracy(model: tl.HookedTransformer, device: TorchDevice, test_loader: DataLoader, learned_rotation: Union[Module, Tensor]):
    """
    Evaluates the accuracy of the learned transformation by comparing the predicted embeddings to the actual French embeddings.

    Args:
        model: The transformer model used for evaluation.
        device: The device on which the model is allocated.
        test_loader: DataLoader for the testing dataset.
        learned_rotation: The learned transformation to be evaluated.

    Returns:
        None. Prints the accuracy of the learned transformation.
    """
    correct_count = 0
    total_count = 0
    for batch in test_loader:
        if len(batch) == 4:
            en_embed, fr_embed, en_attn_mask, fr_attn_mask = batch
        else:
            en_embed, fr_embed = batch
            en_attn_mask, fr_attn_mask = None, None

        en_embed = en_embed.to(device)
        fr_embed = fr_embed.to(device)
        pred = word_pred_from_embeds(en_embed, learned_rotation)

        for i in range(len(en_embed)):
            logits = einsum(en_embed[i], model.embed.W_E, "d_model, vocab d_model -> vocab")
            en_str = model.to_single_str_token(logits.argmax().item())
            logits = einsum(fr_embed[i], model.embed.W_E, "d_model, vocab d_model -> vocab")
            fr_str = model.to_single_str_token(logits.argmax().item())
            logits = einsum(pred[i], model.embed.W_E, "d_model, vocab d_model -> vocab")
            pred_str = model.to_single_str_token(logits.argmax().item())
            if correct := (fr_str == pred_str):
                correct_count += 1
            print("English:", en_str, "French:", fr_str)
            print("English to French rotation", "✅" if correct else "❌")
            get_most_similar_embeddings(
                model,
                pred[i],
                top_k=4,
                apply_embed=True,
            )
        total_count += len(en_embed)
    print()
    print("Correct percentage:", correct_count / total_count * 100)

def calc_cos_sim_acc(test_loader: DataLoader, rotation: Union[Module, Tensor]) -> float:
    """
    Calculates the cosine similarity accuracy between predicted and actual embeddings.

    Args:
        test_loader: DataLoader for the testing dataset.
        rotation: The learned transformation to be evaluated.

    Returns:
        The mean cosine similarity accuracy.
    """
    cosine_sims = []
    for batch_idx, (en_embed, fr_embed) in enumerate(test_loader):
        en_embed.to(device)
        fr_embed.to(device)
        pred = word_pred_from_embeds(en_embed, rotation)
        cosine_sim = word_distance_metric(pred, fr_embed)
        cosine_sims.append(cosine_sim)
    return t.cat(cosine_sims).mean().item()

# %% ----------------------- functions --------------------------
def read_file_lines(file_path: str, lines_count: int = 5000) -> list:
    """
    Reads the specified number of lines from a file, excluding the first line.
    Each line read is concatenated with the next line, separated by a space.

    Args:
        file_path: The path to the file to be read.
        lines_count: The number of lines to read from the file.

    Returns:
        A list of concatenated lines read from the file.
    """
    with open(file_path, "r") as file:
        return [file.readline().strip() + " " + file.readline().strip() for _ in range(lines_count + 1)][1:]

def tokenize_texts(model: tl.HookedTransformer, texts: List[str], padding_side: str = "right") -> Tuple[t.Tensor, t.Tensor]:
    """
    Tokenizes a list of texts using the model's tokenizer, with specified padding, and returns the tokenized texts and their attention masks.

    Args:
        model (tl.HookedTransformer): The transformer model used for tokenization.
        texts (List[str]): A list of texts to be tokenized.
        padding_side (str, optional): The side on which to pad the tokenized texts. Defaults to "right".

    Returns:
        Tuple[t.Tensor, t.Tensor]: A tuple containing the tokenized texts as input IDs and their corresponding attention masks.
    """
    model.tokenizer.padding_side = padding_side  # type: ignore
    tokenized = model.tokenizer(texts, padding=True, return_tensors="pt")  # type: ignore
    return tokenized["input_ids"], tokenized["attention_mask"]

def run_and_gather_acts(model: tl.HookedTransformer, dataloader: DataLoader, layers: List[int]) -> Tuple[defaultdict, defaultdict]:
    """
    Runs the model on batches of English and French text embeddings from the dataloader and gathers embeddings from specified layers.

    Args:
        model (tl.HookedTransformer): The transformer model used for gathering activations.
        dataloader (DataLoader): The dataloader containing batches of English and French text embeddings.
        layers (List[int]): A list of integers specifying the layers from which to gather embeddings.

    Returns:
        Tuple[defaultdict, defaultdict]: Two defaultdicts containing lists of embeddings for English and French texts, separated by layer.
    """
    en_embeds, fr_embeds = defaultdict(list), defaultdict(list)
    for en_batch, fr_batch, en_attn_mask, fr_attn_mask in tqdm(dataloader):
        with t.inference_mode():
            _, en_cache = model.run_with_cache(en_batch, prepend_bos=True)
            _, fr_cache = model.run_with_cache(fr_batch, prepend_bos=True)
            for layer in layers:
                en_resids = en_cache[f"blocks.{layer}.hook_resid_pre"].flatten(start_dim=0, end_dim=1)
                fr_resids = fr_cache[f"blocks.{layer}.hook_resid_pre"].flatten(start_dim=0, end_dim=1)
                en_mask_flat = en_attn_mask.flatten(start_dim=0, end_dim=1)
                fr_mask_flat = fr_attn_mask.flatten(start_dim=0, end_dim=1)
                en_resids_filtered = en_resids[en_mask_flat == 1].detach().clone().cpu()
                fr_resids_filtered = fr_resids[fr_mask_flat == 1].detach().clone().cpu()
                en_embeds[layer].append(en_resids_filtered)
                fr_embeds[layer].append(fr_resids_filtered)
    return en_embeds, fr_embeds

def save_acts(cache_folder: str, filename_base: str, en_resids: Dict[int, t.Tensor], fr_resids: Dict[int, t.Tensor]):
    """
    Saves model activations, separated by layer, to the specified cache folder.

    Args:
        cache_folder (str): The folder path where the activations will be saved.
        filename_base (str): The base name for the saved files.
        en_resids (Dict[int, t.Tensor]): A dictionary containing lists of English embeddings, separated by layer.
        fr_resids (Dict[int, t.Tensor]): A dictionary containing lists of French embeddings, separated by layer.

    """
    en_layers = [layer for layer in en_acts]
    fr_layers = [layer for layer in fr_acts]
    t.save(en_resids, f"{cache_folder}/{filename_base}-en-layers-{en_layers}.pt")
    t.save(fr_resids, f"{cache_folder}/{filename_base}-fr-layers-{fr_layers}.pt")

# -------------- functions 3 - train fr en embed rotation ------------------
def mean_vec(train_en_resids, train_fr_resids):
    return train_en_resids.mean(dim=0) - train_fr_resids.mean(dim=0)

def perform_translation_tests(model, en_file, fr_file, layer_idx, gen_length):
    """
    Performs translation tests on a model by generating translations for a set of English and French strings.
    It reads the last 1000 lines from both English and French files, appending two lines at a time into test strings.
    For each pair of test strings, it prints the original string, generates a translation by iteratively appending the most likely next token,
    and prints the generated translation. For French strings, it temporarily modifies the model's behavior using a `steering_hook` during translation.

    Args:
        model: The transformer model used for translation tests.
        en_file: The file containing English strings.
        fr_file: The file containing French strings.
        layer_idx: The index of the layer to apply the steering hook.
        gen_length: The number of tokens to generate for the translation.

    Returns:
        None. Performs translation tests and prints the results.
    """
    # Load and prepare test strings
    test_en_strs = load_test_strings(en_file, skip_lines=1000)
    test_fr_strs = load_test_strings(fr_file, skip_lines=1000)

    # Perform translation tests
    for idx, (test_en_str, test_fr_str) in enumerate(zip(test_en_strs, test_fr_strs)):
        print("\n----------------------------------------------")

        generate_translation(model, test_en_str, gen_length)
        generate_translation_with_hook(model, test_fr_str, gen_length, layer_idx)
        
        if idx > 5:
            break

def load_test_strings(file_path, skip_lines):
    """
    Loads test strings from a file, skipping the first `skip_lines` lines.

    Args:
        file_path: The path to the file from which to load test strings.
        skip_lines: The number of lines to skip before loading test strings.

    Returns:
        A list of test strings loaded from the file.
    """
    test_strs = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= skip_lines:
                next_line = next(f, '').strip()
                test_strs.append(line.strip() + " " + next_line)
    return test_strs

def generate_translation(model, test_str, gen_length):
    """
    Generates a translation for a given string using the model.

    Args:
        model: The transformer model used for generating translations.
        test_str: The string to translate.
        gen_length: The number of tokens to generate for the translation.

    Returns:
        The generated translation.
    """
    print("Original:", test_str)
    original_len = len(test_str)
    for _ in range(gen_length):
        top_tok = model(test_str, prepend_bos=True)[:, -1].argmax(dim=-1)
        top_tok_str = model.to_string(top_tok)
        test_str += top_tok_str
    print("Translation:", test_str[original_len:])
    return test_str

def generate_translation_with_hook(model, test_str, gen_length, layer_idx):
    """
    Generates a translation for a given string using the model with a steering hook.

    Args:
        model: The transformer model used for generating translations.
        test_str: The string to translate.
        gen_length: The number of tokens to generate for the translation.
        layer_idx: The index of the layer to apply the steering hook.

    Returns:
        The generated translation with the steering hook applied.
    """
    print("Original:", test_str)
    original_len = len(test_str)
    with remove_hooks() as handles, t.inference_mode():
        handle = model.blocks[layer_idx].hook_resid_pre.register_forward_hook(steering_hook)
        handles.add(handle)
        for _ in range(gen_length):
            top_tok = model(test_str, prepend_bos=True)[:, -1].argmax(dim=-1)
            top_tok_str = model.to_string(top_tok)
            test_str += top_tok_str
    print("Translation:", test_str[original_len:])
    return test_str

def steering_hook(
    module: t.nn.Module, input: Tuple[t.Tensor], output: t.Tensor
) -> t.Tensor:
    prefix_toks, final_tok = input[0][:, :-1], input[0][:, -1]
    # layernormed_final_tok = layer_norm(final_tok, [d_model])
    # rotated_final_tok = pred_from_embeds(layernormed_final_tok)
    rotated_final_tok = word_pred_from_embeds(final_tok, learned_rotation)
    # rotated_final_tok = fr_to_en_mean_vec + layernormed_final_tok
    # rotated_final_tok = fr_to_en_mean_vec + final_tok
    # rotated_final_tok = t.zeros_like(rotated_final_tok)
    out = t.cat([prefix_toks, rotated_final_tok.unsqueeze(1)], dim=1)
    return out

# %% -------------- joseph experiment - learn rotation ------------------ 
en_toks, fr_toks = generate_tokens(model, n_toks, device)
en_embeds, fr_embeds = generate_embeddings(model, en_toks, fr_toks)
train_loader, test_loader = create_data_loaders(en_embeds, fr_embeds, train_ratio=0.99, batch_size=512)
initial_rotation, optim = initialize_transform_and_optim(d_model, device, "rotation", lr=0.0002)
learned_rotation = train_and_evaluate(model, device, train_loader, test_loader, initial_rotation, optim, 50)
print("Test Accuracy:", calc_cos_sim_acc(test_loader, learned_rotation))

# %% ------------ joseph experiment - generate fr en bloom embed data for europarl dataset -------

en_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.en"
fr_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.fr"
batch_size = 2
layers = [20, 25, 27, 29]

# Read the first 5000 lines of the files (excluding the first line)
with open(en_file, "r") as f:
    en_strs_list = [f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(5001)][1:]
with open(fr_file, "r") as f:
    fr_strs_list = [f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(5001)][1:]

en_toks, en_attn_mask = tokenize_texts(model, en_strs_list)
fr_toks, fr_attn_mask = tokenize_texts(model, fr_strs_list)

train_loader = create_data_loaders(en_toks, fr_toks, batch_size=2, en_attn_mask=en_attn_mask, fr_attn_mask=fr_attn_mask)
en_acts, fr_acts = run_and_gather_acts(model, train_loader, layers)

cache_folder = repo_path_to_abs_path(".activation_cache")
filename_base = "bloom-3b"
save_acts(cache_folder, filename_base, en_acts, fr_acts)

# %% ----------- joseph experiment - load fr en embed data -------------------------------

en_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.en"
fr_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.fr"

train_en_resids = t.load(
    f"{repo_path_to_abs_path('.activation_cache')}/europarl_v7_fr_en_double_prompt_all_toks"
    + f"-{model.cfg.model_name}-lyrs_[20, 25, 27, 29]-en.pt"
)
train_fr_resids = t.load(
    f"{repo_path_to_abs_path('.activation_cache')}/europarl_v7_fr_en_double_prompt_all_toks"
    + f"-{model.cfg.model_name}-lyrs_[20, 25, 27, 29]-fr.pt"
)

# %% ------------ joseph experiment - train fr en embed rotation ----------------------

layer_idx = 20
gen_length = 20

fr_to_en_data_loader = create_data_loaders(train_en_resids[layer_idx], train_fr_resids[layer_idx], batch_size=512, match_dims=True)
fr_to_en_mean_vec = mean_vec(train_en_resids[layer_idx], train_fr_resids[layer_idx])

perform_translation_tests(model, en_file, fr_file, layer_idx, gen_length)

# %%# ----------- experiment 1 - perform joseph's experiment but for first 4 layers - learn rotation -------------- 
en_toks, fr_toks = generate_tokens(model, n_toks, device)
en_embeds, fr_embeds = generate_embeddings(model, en_toks, fr_toks)
train_loader, test_loader = create_data_loaders(en_embeds, fr_embeds, train_ratio=0.99, batch_size=512)
initial_rotation, optim = initialize_transform_and_optim(d_model, device, "rotation", lr=0.0002)
learned_rotation = train_and_evaluate(model, device, train_loader, test_loader, initial_rotation, optim, 50)
print("Test Accuracy:", calc_cos_sim_acc(test_loader, learned_rotation))
# %% ------------ experiment 1 - generate fr en bloom embed data for europarl dataset -------

en_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.en"
fr_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.fr"
batch_size = 2
layers = [1, 2, 3, 4]

# Read the first 5000 lines of the files (excluding the first line)
with open(en_file, "r") as f:
    en_strs_list = [f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(5001)][1:]
with open(fr_file, "r") as f:
    fr_strs_list = [f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(5001)][1:]

en_toks, en_attn_mask = tokenize_texts(model, en_strs_list)
fr_toks, fr_attn_mask = tokenize_texts(model, fr_strs_list)

train_loader = create_data_loaders(en_toks, fr_toks, batch_size=2, en_attn_mask=en_attn_mask, fr_attn_mask=fr_attn_mask)
en_acts, fr_acts = run_and_gather_acts(model, train_loader, layers)

cache_folder = repo_path_to_abs_path(".activation_cache")
filename_base = "bloom-3b"
save_acts(cache_folder, filename_base, en_acts, fr_acts)

# %% ----------- joseph experiment - load fr en embed data -------------------------------

en_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.en"
fr_file = "/workspace/auto-circuit/europarl/europarl-v7.fr-en.fr"

train_en_resids = t.load(
    f"{repo_path_to_abs_path('.activation_cache')}/europarl_v7_fr_en_double_prompt_all_toks"
    + f"-{model.cfg.model_name}-lyrs_[20, 25, 27, 29]-en.pt"
)
train_fr_resids = t.load(
    f"{repo_path_to_abs_path('.activation_cache')}/europarl_v7_fr_en_double_prompt_all_toks"
    + f"-{model.cfg.model_name}-lyrs_[20, 25, 27, 29]-fr.pt"
)

# %% ------------ joseph experiment - train fr en embed rotation ----------------------

layer_idx = 20
gen_length = 20

fr_to_en_data_loader = create_data_loaders(train_en_resids[layer_idx], train_fr_resids[layer_idx], batch_size=512, match_dims=True)
fr_to_en_mean_vec = mean_vec(train_en_resids[layer_idx], train_fr_resids[layer_idx])

perform_translation_tests(model, en_file, fr_file, layer_idx, gen_length)

# %% ----------- google 10k text dataset generate acts ------------------------
with open("/workspace/auto-circuit/auto_circuit/google-10000-english.txt", "r") as file:
    en_file= file.read().splitlines()
en_strs_list, fr_strs_list = generate_google_words(model, 9000, device, en_file)

en_toks, en_attn_mask = tokenize_texts(model, en_strs_list)
fr_toks, fr_attn_mask = tokenize_texts(model, fr_strs_list)

train_loader, test_loader = create_data_loaders(en_toks, fr_toks, batch_size=128, train_ratio=0.99, en_attn_mask=en_attn_mask, fr_attn_mask=fr_attn_mask)
cache_folder = repo_path_to_abs_path(".activation_cache")
en_acts, fr_acts = run_and_gather_acts(model, train_loader, layers=[1,2])
filename_base = "bloom-3b"

save_acts(cache_folder, filename_base, en_acts, fr_acts)

# %% ----------- google 10k train fr en rotation ------------------------

train_en_resids = t.load(f"{repo_path_to_abs_path('.activation_cache')}/bloom-3b-en-layer-1.pt")
train_fr_resids = t.load(f"{repo_path_to_abs_path('.activation_cache')}/bloom-3b-fr-layer-1.pt")

layer_idx = 1
gen_length = 20

fr_to_en_data_loader = create_data_loaders(en_embeds, train_en_resids[layer_idx], train_fr_resids[layer_idx], match_dims=True)
fr_to_en_mean_vec = mean_vec(model, train_en_resids[layer_idx], train_fr_resids[layer_idx])

perform_translation_tests(model, en_file, fr_file, layer_idx, gen_length)
