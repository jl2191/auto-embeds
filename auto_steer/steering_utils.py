from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import plotly.express as px
import torch as t
import transformer_lens as tl
from einops import einsum
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from word2word import Word2word
from jaxtyping import Float, Int

from auto_steer.utils.custom_tqdm import tqdm
from auto_steer.utils.misc import (
    get_most_similar_embeddings,
    remove_hooks,
)


def generate_tokens(
    model: tl.HookedTransformer,
    n_toks: int,
    device: Optional[Union[str, t.device]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Generates and translates tokens from English to French.

    Processes a specified number of tokens, translating valid ones from English
    to French, and returns their indices.

    Args:
        model: Transformer model for token processing.
        n_toks: Number of tokens to process.
        device: Device for tensor allocation, defaults to GPU if available.

    Returns:
        Tuple of tensors with indices of valid English and French tokens.
    """
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    en2fr = Word2word("en", "fr")
    en_toks, fr_toks = [], []
    for tok in range(n_toks):
        en_tok_str = model.to_string([tok])
        if len(en_tok_str) < 7 or en_tok_str[0] != " ":
            continue
        try:
            fr_tok_str = " " + en2fr(en_tok_str[1:], n_best=1)[0]
        except Exception as e:
            print(f"Translation failed for {en_tok_str}: {e}")
            continue
        if en_tok_str.lower() == fr_tok_str.lower():  # type: ignore
            continue
        try:
            fr_tok = model.to_single_token(fr_tok_str)
        except Exception as e:
            print(f"Token conversion failed for {fr_tok_str}: {e}")
            continue
        en_toks.append(tok)
        fr_toks.append(fr_tok)
    return t.tensor(en_toks, device=device), t.tensor(fr_toks, device=device)


def generate_google_words(
    model: tl.HookedTransformer,
    n_toks: int,
    en_file: List[str],
    device: Optional[Union[str, t.device]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Takes in a list of English strings then returns an English and French list. The
    French list contains French translations of English words for which a translation
    can be found.

    Args:
        model: The transformer model used for token processing.
        n_toks: The number of tokens to process.
        device: The device on which tensors will be allocated.
        en_file: List of English strings to be translated.

    Returns:
        A tuple of lists containing English strings and their French translations.
    """
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
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


def generate_embeddings(
    model: tl.HookedTransformer,
    en_toks: Tensor,
    fr_toks: Tensor,
    device: Optional[Union[str, t.device]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Generates embeddings for English and French tokens.

    Args:
        model: The transformer model used for generating embeddings.
        en_toks: Tensor of English token indices.
        fr_toks: Tensor of French token indices.
        device: The device on which tensors will be allocated.

    Returns:
        A tuple of tensors containing embeddings for English and French tokens.
    """
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    en_embeds = model.embed.W_E[en_toks].detach().clone().to(device)
    fr_embeds = model.embed.W_E[fr_toks].detach().clone().to(device)
    return en_embeds, fr_embeds


def create_data_loaders(
    en_embeds: Tensor,
    fr_embeds: Tensor,
    batch_size: int,
    train_ratio: float = 1.0,
    en_attn_mask: Optional[Tensor] = None,
    fr_attn_mask: Optional[Tensor] = None,
    match_dims: bool = False,
    mask: bool = False,
) -> Union[
    DataLoader[Tuple[Tensor, ...]],
    Tuple[DataLoader[Tuple[Tensor, ...]], DataLoader[Tuple[Tensor, ...]]],
]:
    """
    Refactored function to create data loaders for training and optionally testing
    datasets from embedding tensors and attention masks, with an option to match
    dimensions and apply masks.

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
        A DataLoader for the training dataset, and optionally a DataLoader for the
        testing dataset.
    """
    # Match dimensions if required
    if match_dims:
        min_len = min(len(en_embeds), len(fr_embeds))
        en_embeds, fr_embeds = en_embeds[:min_len], fr_embeds[:min_len]
        if mask and en_attn_mask is not None and fr_attn_mask is not None:
            min_len_mask = min(len(en_attn_mask), len(fr_attn_mask))
            en_attn_mask, fr_attn_mask = (
                en_attn_mask[:min_len_mask],
                fr_attn_mask[:min_len_mask],
            )

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
        print(f"Train size: {train_size}, Test size: {test_size}")
        train_set, test_set = t.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        train_loader = t.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_loader = t.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=True
        )
        return train_loader, test_loader
    else:
        train_loader = t.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        return train_loader


def initialize_transform_and_optim(
    d_model: int,
    transformation: str,
    lr: float,
    device: Optional[Union[str, t.device]] = None,
    train_en_resids: Optional[Tensor] = None,
    train_fr_resids: Optional[Tensor] = None,
) -> Tuple[Union[Module, Tensor], Optimizer]:
    """
    Initializes a transformation and its corresponding optimizer based on the specified
    transformation type.

    Args:
        d_model: The dimensionality of the model embeddings.
        device: The device on which the transformation will be allocated.
        transformation: The type of transformation to initialize.
        lr: Learning rate for the optimizer.
        train_en_resids: The mean difference between English and French residuals.
        train_fr_resids: The mean difference between English and French residuals.

    Returns:
        A tuple containing the transformation and its optimizer.
    """
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    if transformation == "empty_translation":
        transform = t.zeros([d_model], device=device, requires_grad=True)
        optim = t.optim.Adam([transform], lr=0.0002)
    if transformation == "mean_translation":
        if train_en_resids is None or train_fr_resids is None:
            raise ValueError(
                "English and French residuals must be provided for \
                             mean-centered steering transformation."
            )
        mean_diff = train_en_resids.mean(dim=0) - train_fr_resids.mean(dim=0)
        transform = mean_diff.to(device)
        optim = t.optim.Adam([transform], lr=lr)
    elif transformation == "rotation":
        transform = t.nn.Linear(d_model, d_model, bias=False, device=device)
        # optim = t.optim.Adam(list(learned_rotation.parameters()) + [translate],
        # lr=0.0002)
        optim = t.optim.Adam(transform.parameters(), lr=lr)
    elif transformation == "linear_map":
        initial_rotation = t.nn.Linear(d_model, d_model, bias=False, device=device)
        transform = t.nn.utils.parametrizations.orthogonal(initial_rotation, "weight")
        optim = t.optim.Adam(list(transform.parameters()), lr=0.0002)
        # optim = t.optim.Adam(list(linear_map.parameters()) + [translate], lr=0.01)
    else:
        raise Exception("the supplied transform was unrecognized")
    return transform, optim


def word_pred_from_embeds(
    embeds: Tensor, transformation: Union[Module, Tensor], lerp: float = 1.0
) -> Tensor:
    """
    Applies a specified transformation to the input embeddings.

    Args:
        embeds (Tensor): The input embeddings to be transformed.
        transformation (Union[Module, Tensor]): The transformation to be applied.
        Can be a PyTorch module or a tensor.
        lerp (float, optional): Linear interpolation factor. Defaults to 1.0, meaning
        full rotation is applied.

    Returns:
        Tensor: The transformed embeddings after applying the transformation.
    """

    if isinstance(transformation, t.nn.Module):
        return transformation(embeds)
    else:  # transformation is a Tensor
        return embeds @ transformation


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


def train_and_evaluate_transform(
    model: tl.HookedTransformer,
    train_loader: DataLoader[Tuple[Tensor, ...]],
    test_loader: DataLoader[Tuple[Tensor, ...]],
    initial_rotation: Union[Module, Tensor],
    optim: Optimizer,
    n_epochs: int,
    device: Optional[Union[str, t.device]] = None,
) -> Union[Module, Tensor]:
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
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    loss_history = []
    initial_rotation.to(
        device
    )  # Ensure the learned_rotation model is on the correct device
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch_idx, (en_embed, fr_embed) in enumerate(train_loader):
            en_embed = en_embed.to(device)
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
    evaluate_accuracy(model, test_loader, learned_rotation, device)
    return learned_rotation


def evaluate_accuracy(
    model: tl.HookedTransformer,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    learned_rotation: Union[Module, Tensor],
    device: Optional[Union[str, t.device]] = None,
):
    """
    Evaluates the accuracy of the learned transformation by comparing the predicted
    embeddings to the actual French embeddings.

    Args:
        model: The transformer model used for evaluation.
        device: The device on which the model is allocated.
        test_loader: DataLoader for the testing dataset.
        learned_rotation: The learned transformation to be evaluated.

    Returns:
        None. Prints the accuracy of the learned transformation.
    """
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    correct_count = 0
    total_count = 0
    for batch in test_loader:
        if len(batch) == 4:
            en_embed, fr_embed, en_attn_mask, fr_attn_mask = batch
        else:
            en_embed, fr_embed = batch

        en_embed = en_embed.to(device)
        fr_embed = fr_embed.to(device)
        pred = word_pred_from_embeds(en_embed, learned_rotation)

        for i in range(len(en_embed)):
            logits = einsum(
                en_embed[i], model.embed.W_E, "d_model, vocab d_model -> vocab"
            )
            en_str = model.to_single_str_token(logits.argmax().item())
            logits = einsum(
                fr_embed[i], model.embed.W_E, "d_model, vocab d_model -> vocab"
            )
            fr_str = model.to_single_str_token(logits.argmax().item())
            logits = einsum(pred[i], model.embed.W_E, "d_model, vocab d_model -> vocab")
            pred_str = model.to_single_str_token(logits.argmax().item())  # type: ignore
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


def calc_cos_sim_acc(
    test_loader: DataLoader[Tuple[Tensor, ...]], rotation: Union[Module, Tensor], device: Optional[str] = None
) -> float:
    """
    Calculates the cosine similarity accuracy between predicted and actual embeddings.

    Args:
        test_loader: DataLoader for the testing dataset.
        rotation: The learned transformation to be evaluated.
        device: The device to perform calculations on.

    Returns:
        The mean cosine similarity accuracy.
    """
    if device is not None:
        rotation.to(device)
    cosine_sims = []
    for batch_idx, (en_embed, fr_embed) in enumerate(test_loader):
        en_embed = en_embed.to(device)
        fr_embed = fr_embed.to(device)
        pred = word_pred_from_embeds(en_embed, rotation)
        cosine_sim = word_distance_metric(pred, fr_embed)
        cosine_sims.append(cosine_sim)
    return t.cat(cosine_sims).mean().item()


# %% ----------------------- functions --------------------------


def read_file_lines(file_path: Union[str, Path], lines_count: int = 5000) -> List[str]:
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
        return [
            file.readline().strip() + " " + file.readline().strip()
            for _ in range(lines_count + 1)
        ][1:]

def tokenize_texts(
    model: tl.HookedTransformer,
    *text_lists: List[str],
    padding_side: str = "right",
    pad_to_same_length: bool = False,
    padding_strategy: str = "longest"
) -> List[Tuple[t.Tensor, t.Tensor]]:
    """
    Tokenizes multiple lists of texts using the model's tokenizer. Optional arguments
    for padding size as well as having all tensors in each of the lists be tokenized and
    padded to the same length. Returns a list of tuples, each containing a single
    tensor for input IDs and a single tensor for attention masks for each text list.
    The padding_strategy argument is ignored if pad_to_same_length is set to True.

    Args:
        model (tl.HookedTransformer): The transformer model for tokenization.
        *text_lists (List[str]): Variable number of text lists to be tokenized.
        padding_side (str, optional): The side for padding tokenized texts.
                                      Defaults to "right".
        pad_to_same_length (bool, optional): If True, pads all tokenized text lists
                                             to the same length. Defaults to False.
        padding_strategy (str, optional): The strategy for padding tokenized texts.
                                          Defaults to "longest".

    Returns:
        List[Tuple[t.Tensor, t.Tensor]]: A list of tuples, each containing a single
        tensor of tokenized texts as input IDs and a single tensor for their
        corresponding attention masks, respectively, for each text list.
    """
    original_padding_side = model.tokenizer.padding_side  # type: ignore
    model.tokenizer.padding_side = padding_side
    tokenized_results = []

    padding = padding_strategy if not pad_to_same_length else "longest"

    for text_list in text_lists:
        tokenized = model.tokenizer(text_list, padding=padding, return_tensors="pt")  # type: ignore
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
        tokenized_results.append((input_ids, attention_mask))

    if pad_to_same_length:
        max_length_all_lists = 0
        # calculate the max token sequence length across all lists
        for tokenized_list in tokenized_results:
            input_ids, _ = tokenized_list
            lengths = [len(tensor) for tensor in input_ids]
            max_length_this_list = max(lengths)
            max_length_all_lists = max(max_length_all_lists, max_length_this_list)
        # rerun the tokenizer for all lists, this time with the found max length
        tokenized_results = []
        for text_list in text_lists:
            tokenized = model.tokenizer(text_list,
                                        padding="max_length",
                                        max_length=max_length_all_lists,
                                        return_tensors="pt")  # type: ignore

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            tokenized_results.append((input_ids, attention_mask))
    # set the model's tokenizer padding side to what it was before this function call
    model.tokenizer.padding_side = original_padding_side  # type: ignore
    return tokenized_results


def run_and_gather_acts(
    model: tl.HookedTransformer,
    dataloader: DataLoader[Tuple[Tensor, ...]],
    layers: List[int],
) -> Tuple[Dict[int, List[Tensor]], Dict[int, List[Tensor]]]:
    """
    Runs the model on batches of English and French text embeddings from the dataloader
    and gathers embeddings from specified layers.

    Args:
        model: The transformer model used for gathering activations.
        dataloader: The dataloader with batches of English and French text embeddings.
        layers: List of integers specifying the layers for gathering embeddings.

    Returns:
        Two dicts containing lists of embeddings for English and French texts,
        separated by layer.
    """
    en_embeds, fr_embeds = defaultdict(list), defaultdict(list)
    for en_batch, fr_batch, en_attn_mask, fr_attn_mask in tqdm(dataloader):
        with t.inference_mode():
            _, en_cache = model.run_with_cache(en_batch, prepend_bos=True)
            _, fr_cache = model.run_with_cache(fr_batch, prepend_bos=True)
            for layer in layers:
                en_resids = en_cache[f"blocks.{layer}.hook_resid_pre"]
                en_resids_flat = en_resids.flatten(start_dim=0, end_dim=1)
                en_mask_flat = en_attn_mask.flatten(start_dim=0, end_dim=1)
                filtered_en_resids = en_resids_flat[en_mask_flat == 1]
                en_embeds[layer].append(filtered_en_resids.detach().clone().cpu())

                fr_resids = fr_cache[f"blocks.{layer}.hook_resid_pre"]
                fr_resids_flat = fr_resids.flatten(start_dim=0, end_dim=1)
                fr_mask_flat = fr_attn_mask.flatten(start_dim=0, end_dim=1)
                filtered_fr_resids = fr_resids_flat[fr_mask_flat == 1]
                fr_embeds[layer].append(filtered_fr_resids.detach().clone().cpu())
    en_embeds = dict(en_embeds)
    fr_embeds = dict(fr_embeds)
    return en_embeds, fr_embeds


def save_acts(
    cache_folder: Union[str, Path],
    filename_base: str,
    en_acts: Dict[int, List[t.Tensor]],
    fr_acts: Dict[int, List[t.Tensor]],
):
    """
    Saves model activations, separated by layer, to the specified cache folder.

    Args:
        cache_folder: The folder path where the activations will be saved.
        filename_base : The base name for the saved files.
        en_acts: A dict containing lists of English embeddings, separated by layer.
        fr_acts: A dict containing lists of French embeddings, separated by layer.

    """
    en_layers = [layer for layer in en_acts]
    fr_layers = [layer for layer in fr_acts]
    t.save(en_acts, f"{cache_folder}/{filename_base}-en-layers-{en_layers}.pt")
    t.save(fr_acts, f"{cache_folder}/{filename_base}-fr-layers-{fr_layers}.pt")


# -------------- functions 3 - train fr en embed rotation ------------------
def mean_vec(train_en_resids: t.Tensor, train_fr_resids: t.Tensor) -> t.Tensor:
    """
    Calculates the mean vector difference between English and French residuals.

    Args:
        train_en_resids (t.Tensor): The tensor containing English residuals.
        train_fr_resids (t.Tensor): The tensor containing French residuals.

    Returns:
        t.Tensor: The mean vector difference between English and French residuals.
    """
    return train_en_resids.mean(dim=0) - train_fr_resids.mean(dim=0)


def perform_translation_tests(
    model: t.nn.Module,
    en_strs: List[str],
    fr_strs: List[str],
    layer_idx: int,
    gen_length: int,
    transformation: Union[Module, Tensor],
) -> None:
    """
    Performs translation tests on a model by generating translations for English
    and French strings. For each pair of strings in the provided lists, it prints the
    original string, generates a translation by iteratively appending the most likely
    next token, and prints the generated translation. For French strings, it modifies
    the model's behavior using a `steering_hook` during translation.

    Args:
        model (t.nn.Module): The transformer model used for translation tests.
        en_strs (List[str]): The list containing English strings.
        fr_strs (List[str]): The list containing French strings.
        layer_idx (int): The index of the layer to apply the steering hook.
        gen_length (int): The number of tokens to generate for the translation.
    """

    # Perform translation tests
    for idx, (test_en_str, test_fr_str) in enumerate(zip(en_strs, fr_strs)):
        print("\n----------------------------------------------")

        generate_translation(model, test_en_str, gen_length)
        generate_translation_with_hook(
            model, test_fr_str, gen_length, layer_idx, transformation
        )

        if idx > 5:
            break


def load_test_strings(file_path: Union[str, Path], skip_lines: int) -> List[str]:
    """
    Loads test strings from a file, skipping the first `skip_lines` lines.

    Args:
        file_path (str): The path to the file from which to load test strings.
        skip_lines (int): The number of lines to skip before loading test strings.

    Returns:
        List[str]: A list of test strings loaded from the file.
    """
    test_strs = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= skip_lines:
                next_line = next(f, "").strip()
                test_strs.append(line.strip() + " " + next_line)
    return test_strs


def generate_translation(model: t.nn.Module, test_str: str, gen_length: int) -> str:
    """
    Generates a translation for a given string using the model.

    Args:
        model (t.nn.Module): The transformer model used for generating translations.
        test_str (str): The string to translate.
        gen_length (int): The number of tokens to generate for the translation.

    Returns:
        str: The generated translation.
    """
    print("test_en_str:", test_str)
    original_len = len(test_str)
    for _ in range(gen_length):
        top_tok = model(test_str, prepend_bos=True)[:, -1].argmax(dim=-1)
        top_tok_str = model.to_string(top_tok)
        test_str += top_tok_str
    print("result fr str:", test_str[original_len:])
    return test_str


def generate_translation_with_hook(
    model: t.nn.Module,
    test_str: str,
    gen_length: int,
    layer_idx: int,
    transformation: Union[Module, Tensor],
) -> str:
    """
    Generates a translation for a given string using the model with a steering hook.

    Args:
        model (t.nn.Module): The transformer model used for generating translations.
        test_str (str): The string to translate.
        gen_length (int): The number of tokens to generate for the translation.
        layer_idx (int): The index of the layer to apply the steering hook.
        transformation (Union[Module, Tensor]): The transformation to apply using the
        steering hook.

    Returns:
        str: The generated translation with the steering hook applied.
    """
    print("test_fr_str:", test_str)
    original_len = len(test_str)
    with remove_hooks() as handles, t.inference_mode():
        handle = model.blocks[layer_idx].hook_resid_pre.register_forward_hook(
            lambda module, input, output: steering_hook(
                module, input, output, transformation
            )
        )
        handles.add(handle)
        for _ in range(gen_length):
            top_tok = model(test_str, prepend_bos=True)[:, -1].argmax(dim=-1)
            top_tok_str = model.to_string(top_tok)
            test_str += top_tok_str
    print("result fr str", test_str[original_len:])
    return test_str


def steering_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor],
    output: t.Tensor,
    transformation: Union[Module, Tensor],
) -> t.Tensor:
    """
    Modifies a module's output during translation by applying a transformation.

    Intended for use as a forward hook on a transformer model layer, this function
    steers the model's behavior, such as aligning embeddings across languages.

    Args:
        module (t.nn.Module): The module where the hook is registered.
        input (Tuple[t.Tensor]): Input tensors to the module, with the first tensor
                                 usually being the input embeddings.
        output (t.Tensor): The original output tensor of the module.
        transformation (Union[Module, Tensor]): The transformation to apply to the
                                                output tensor, which could be a
                                                learned matrix or another module.

    Returns:
        t.Tensor: The output tensor after applying the transformation, replacing the
                  original output in the model's forward pass.
    """
    prefix_toks, final_tok = input[0][:, :-1], input[0][:, -1]
    rotated_final_tok = word_pred_from_embeds(final_tok, transformation)
    out = t.cat([prefix_toks, rotated_final_tok.unsqueeze(1)], dim=1)
    return out

# %%
