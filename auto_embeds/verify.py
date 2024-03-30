import random
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import pandas as pd
import plotly.express as px
import torch as t
import torch.nn as nn
import transformer_lens as tl
from plotly.graph_objects import Figure
from rich.table import Table
from scipy import stats
from torch import Tensor
from torch.utils.data import DataLoader

from auto_embeds.data import OtherWords, SelectedWord, VerifyWordPairAnalysis
from auto_embeds.embed_utils import tokenize_word_pairs
from auto_embeds.utils.misc import calculate_gradient_color


def verify_transform(
    model: tl.HookedTransformer,
    transformation: nn.Module,
    test_loader: DataLoader[Tuple[Tensor, ...]],
) -> Dict[str, Any]:
    """
    Evaluates the transformation's effectiveness in translating source language tokens
    to target language tokens by examining the relationship between accuracy and the
    cosine similarity of embeddings. Returns a dictionary containing cosine similarity,
    Euclidean distance, and strings for source, target, and predicted tokens.

    Args:
        model: A transformer model with hooked embeddings and a tokenizer.
        transformation: The transformation module applied to source language embeddings.
        test_loader: DataLoader for the test dataset with source and
                     target language embeddings tuples.

    Returns:
        Dict[str, Any]: A dictionary with keys:
                        - 'cos_sims': Cosine similarities between source and target
                                      language embeddings.
                        - 'euc_dists': Euclidean distances between source and target
                                       language embeddings.
                        - 'en_strs': Strings for source language tokens.
                        - 'fr_strs': Strings for target language tokens.
                        - 'top_pred_strs': Strings for top predicted tokens.
    """

    # creates a dict for the source language tokens in a test dataset that is created
    # using within this confidence check python file (which has all the entries in order
    # of cosine similarity to some randomly chosen word)

    # we are doing this because if we find that accuracy decreases as we get further
    # away in terms of cosine similarity from our randomly chosen word, then this
    # increases the odds that our results are due to our rotation "cheating" having
    # somewhat already seen the word. if we find that instead it remains the same, then
    # this seems to increase the odds that we have found a general source to target
    # language translation transformation

    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    verify_results = {
        "cos_sims": [],
        "euc_dists": [],
        "en_strs": [],
        "fr_strs": [],
        "top_pred_strs": [],
    }
    # for each batch in the test dataset
    with t.no_grad():
        for batch in test_loader:
            # we get the embeddings for the source and target language
            en_embeds, fr_embeds = batch
            top_pred_embeds = transformation(en_embeds)

            en_logits = einops.einsum(
                en_embeds,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
            en_strs: List[str] = model.tokenizer.batch_decode(
                en_logits.squeeze().argmax(dim=-1)
            )
            fr_logits = einops.einsum(
                fr_embeds,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
            fr_strs: List[str] = model.tokenizer.batch_decode(
                fr_logits.squeeze().argmax(dim=-1)
            )
            # transform it using our learned rotation
            with t.no_grad():
                top_pred_embeds = transformation(en_embeds)
            # get the top predicted tokens
            top_pred_logits = einops.einsum(
                top_pred_embeds,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
            top_pred_strs = model.tokenizer.batch_decode(
                top_pred_logits.squeeze().argmax(dim=-1)
            )
            top_pred_strs = [
                item if isinstance(item, str) else item[0] for item in top_pred_strs
            ]
            assert all(isinstance(item, str) for item in top_pred_strs)

            # calculate the cosine similarity and euclidean distance between this
            # predicted token and the actual target token
            cos_sims = t.cosine_similarity(top_pred_embeds, fr_embeds, dim=-1).squeeze(
                1
            )
            # cos_sims should be shape [batch] since we removed the pos dimension
            euc_dists = t.pairwise_distance(top_pred_embeds, fr_embeds).squeeze(1)
            # euc_dists should be shape [batch] since we removed the pos dimension

            verify_results["en_strs"].extend(en_strs)
            verify_results["fr_strs"].extend(fr_strs)
            verify_results["top_pred_strs"].extend(top_pred_strs)
            verify_results["cos_sims"].append(cos_sims)
            verify_results["euc_dists"].append(euc_dists)

    # aggregate results from all batches
    verify_results["cos_sims"] = t.cat(verify_results["cos_sims"])
    verify_results["euc_dists"] = t.cat(verify_results["euc_dists"])

    return verify_results


def verify_transform_table_from_dict(verify_results: Dict[str, Any]) -> Table:
    """
    Returns a rich table comparing cosine similarity and Euclidean distance between
    predicted and actual tokens using the results from a verify_transform dictionary.

    Args:
        verify_results: A dictionary containing results from verify_transform()

    Returns:
        Table: A rich table object with the comparison results.
    """
    table = Table(title="Cosine Similarity and Euclidean Distance Comparisons")
    table.add_column("Rank", style="bold magenta")
    table.add_column("Source Word", justify="right", style="cyan", no_wrap=True)
    table.add_column("Target Word", justify="right", style="green")
    table.add_column("Pred Token", justify="right", style="magenta")
    table.add_column("Cos Sims", justify="right", style="bright_yellow")
    table.add_column("Euc Dist", justify="right", style="bright_yellow")

    cos_sim_values = verify_results["cos_sims"].tolist()
    euc_dist_values = verify_results["euc_dists"].tolist()
    cos_sim_min, cos_sim_max = min(cos_sim_values), max(cos_sim_values)
    euc_dist_min, euc_dist_max = min(euc_dist_values), max(euc_dist_values)

    # Limit the number of displayed results
    for rank, (en_str, fr_str, top_pred_str, cos_sim, euc_dist) in enumerate(
        zip(
            verify_results["en_strs"],
            verify_results["fr_strs"],
            verify_results["top_pred_strs"],
            verify_results["cos_sims"],
            verify_results["euc_dists"],
        ),
        1,
    ):
        cos_sim = cos_sim.item()
        euc_dist = euc_dist.item()

        # Gradient color for cosine similarity
        cos_sim_color = calculate_gradient_color(cos_sim, cos_sim_min, cos_sim_max)
        euc_dist_color = calculate_gradient_color(
            euc_dist, euc_dist_min, euc_dist_max, reverse=True
        )

        en_str_styled = f"[plum3 on grey30]{en_str}[/plum3 on grey30]"
        fr_str_styled = f"[plum3 on grey30]{fr_str}[/plum3 on grey30]"
        if top_pred_str == fr_str:
            top_pred_str_styled = f"[green on grey30]{top_pred_str}[/green on grey30]"
        else:
            top_pred_str_styled = f"[red1 on grey30]{top_pred_str}[/red1 on grey30]"
        cos_sim_styled = f"[{cos_sim_color}]{cos_sim:.4f}[/{cos_sim_color}]"
        euc_dist_styled = f"[{euc_dist_color}]{euc_dist:.4f}[/{euc_dist_color}]"
        table.add_row(
            str(rank),
            en_str_styled,
            fr_str_styled,
            top_pred_str_styled,
            cos_sim_styled,
            euc_dist_styled,
        )

    return table


def calc_tgt_is_closest_embed(
    model: tl.HookedTransformer,
    src_toks: t.Tensor,
    tgt_toks: t.Tensor,
    other_toks: t.Tensor,
    other_embeds: t.Tensor,
    device: Optional[Union[str, t.device]] = None,
) -> Dict[str, Union[str, List[str]]]:
    """
    Calculates the percentage of instances where the target language token is within
    the top 1 and top 5 closest tokens in terms of cosine similarity.

    Args:
        model: The model used for embedding and token decoding.
        src_toks: Source tokens tensor.
        tgt_toks: Target tokens tensor.
        other_toks: Other tokens tensor to compare against.
        other_embeds: Embeddings of other tokens.
        device: The device on which to allocate tensors. If None, defaults to
            model.cfg.device.

    Returns:
        A dictionary containing a summary of the results and detailed results for each
        source token. Keys are ["summary"] and ["details"].
    """
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    if device is None:
        device = model.cfg.device
    other_toks = other_toks.to(device)
    src_toks = src_toks.to(device)
    tgt_toks = tgt_toks.to(device)
    correct_count_top_5 = 0
    correct_count_top_1 = 0
    total_count = len(src_toks)

    details = []

    for i, (src_tok, correct_tgt_tok) in enumerate(zip(src_toks, tgt_toks)):
        # Embed the source token
        src_embed = model.embed.W_E[src_tok].detach().clone().squeeze(0)
        # shape [d_model]

        # Exclude the current source token from other_toks and other_embeds to avoid
        # self-matching
        valid_indices = (other_toks != src_tok).squeeze(-1)
        # the mask valid_indices should be of shape [batch]

        valid_other_toks = other_toks[valid_indices]
        valid_other_embeds = other_embeds[valid_indices]

        # The cosine similarities between the source token and all the other tokens
        cos_sims = t.cosine_similarity(src_embed, valid_other_embeds, dim=-1)

        # Get the indices of the top 5 target tokens with the highest cosine similarity
        # that is valid (i.e. not the same as the source language token)
        top_5_indices = t.topk(cos_sims, 5).indices

        # Check if the correct target token is among the top 5
        is_correct_in_top_5 = correct_tgt_tok in valid_other_toks[top_5_indices]
        is_correct_in_top_1 = correct_tgt_tok == valid_other_toks[top_5_indices[0]]

        if is_correct_in_top_5:
            correct_count_top_5 += 1

        if is_correct_in_top_1:
            correct_count_top_1 += 1
            status = "Correct ✅ "
        else:
            status = "Incorrect ❌"

        src_tok_str = model.tokenizer.decode(src_tok)
        correct_tgt_tok_str = model.tokenizer.decode(correct_tgt_tok)
        top_5_tokens = [
            model.tokenizer.decode(valid_other_toks[index], skip_special_tokens=True)
            for index in top_5_indices
        ]
        cos_sim_values = [cos_sims[index].item() for index in top_5_indices]
        top_5_details = "\n".join(
            f"  {rank}. {token} (Cosine Similarity: {cos_sim:.4f})"
            for rank, (token, cos_sim) in enumerate(zip(top_5_tokens, cos_sim_values))
        )
        detail = (
            f"{i+1} {status}\n"
            f"Source Token: '{src_tok_str}'\n"
            f"Target Token: '{correct_tgt_tok_str}'\n"
            f"Top 5 tokens with highest cosine similarity:\n"
            f"{top_5_details}\n\n"
        )
        details.append(detail)

    # Calculate the percentage where the hypothesis holds true for top 5 and top 1
    percentage_correct_top_1 = (correct_count_top_1 / total_count) * 100
    percentage_correct_top_5 = (correct_count_top_5 / total_count) * 100
    summary = (
        f"Percentage where the hypothesis is true (correct translation is top 1): "
        f"{percentage_correct_top_1:.2f}%\n"
        f"Percentage where the hypothesis is true (correct translation in top 5): "
        f"{percentage_correct_top_5:.2f}%"
    )

    return {"summary": summary, "details": details}


def generate_top_word_pairs_table(
    model: tl.HookedTransformer,
    random_word: str,
    other_toks: t.Tensor,
    top_indices: t.Tensor,
    euc_dists: t.Tensor,
    cos_sims: t.Tensor,
    sort_by: str = "cos_sim",
    display_limit: int = 50,
) -> Table:
    """
    Generates a table of top word pairs based on cosine similarity or Euclidean
    distance, highlighting the relationship between a random word and other tokens.
    Used to eyeball verify the quality of a learned embedding transformation. Example
    usage in experiments/verify.py

    Args:
        model: The model used for token decoding.
        random_word: The random word to compare against other tokens.
        other_toks: Tensor of other tokens to compare.
        top_indices: Indices of top tokens based on the chosen metric.
        distances: Euclidean distances of other tokens from the random word.
        cos_sims: Cosine similarities of other tokens with the random word.
        sort_by: Criterion for sorting the tokens ('cos_sim' or 'euc_dist').
        display_limit: Number of entries to display in the table.

    Returns:
        A rich Table object containing the ranked tokens, their cosine similarity,
        and Euclidean distance to the random word.
    """
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    if sort_by == "cos_sim":
        title_sort_by = "Cosine Similarity"
    elif sort_by == "euc_dist":
        title_sort_by = "Euclidean Distance"
    else:
        raise ValueError("Supported sort functions are 'cos_sim' and 'euc_dist'")

    table = Table(
        show_header=True,
        title=f"Closest tokens in {model.cfg.model_name} to "
        f"[plum3 on grey23]{random_word}[/plum3 on grey23] sorted by {title_sort_by}",
    )
    table.add_column("Rank")
    table.add_column("Token")
    table.add_column("Cos Sim")
    table.add_column("Euc Dist")

    # Determine the range of cosine similarities and euclidean distances for gradient
    # calculation
    cos_sim_values = [cos_sims[index].item() for index in top_indices]
    euc_dist_values = [euc_dists[index].item() for index in top_indices]
    cos_sim_min, cos_sim_max = min(cos_sim_values), max(cos_sim_values)
    euc_dist_min, euc_dist_max = min(euc_dist_values), max(euc_dist_values)

    if sort_by == "cos_sim":
        sorted_indices = sorted(
            top_indices, key=lambda idx: cos_sims[idx].item(), reverse=True
        )
    elif sort_by == "euc_dist":
        sorted_indices = sorted(top_indices, key=lambda idx: euc_dists[idx].item())
    else:
        raise ValueError("Supported sort functions are 'cos_sim' and 'euc_dist'")

    display_start = display_limit // 2
    for rank, index in enumerate(sorted_indices):
        if rank == display_start + 1:
            table.add_row("...", "...", "...", "...")
        elif rank > display_start and rank <= top_indices.shape[0] - display_start:
            continue
        else:
            token = other_toks[index]
            word = model.tokenizer.decode(token)
            cos_sim = cos_sims[index].item()
            euc_dist = euc_dists[index].item()

            # Calculate gradient colors based on the value's magnitude
            cos_sim_color = calculate_gradient_color(cos_sim, cos_sim_min, cos_sim_max)
            euc_dist_color = calculate_gradient_color(
                euc_dist, euc_dist_min, euc_dist_max, reverse=True
            )

            word_styled = f"[plum3 on grey23]{word}[/plum3 on grey23]"
            cos_sim_styled = f"[{cos_sim_color}]{cos_sim:.4f}[/{cos_sim_color}]"
            euc_dist_styled = f"[{euc_dist_color}]{euc_dist:.4f}[/{euc_dist_color}]"
            table.add_row(str(rank), word_styled, cos_sim_styled, euc_dist_styled)

    return table


def plot_cosine_similarity_trend(verify_results: Dict[str, Any]) -> Figure:
    """
    Plots the trend of cosine similarity across ranks with an average trend line.

    Args:
        verify_results: A dictionary containing the cosine similarities, source words,
            target words, and predicted words.

    Returns:
        A Plotly Figure object representing the trend of cosine similarity across ranks.

    """
    cos_sims = verify_results["cos_sims"].tolist()
    ranks = list(range(len(cos_sims)))

    df = pd.DataFrame(
        {
            "Rank": ranks,
            "Cosine Similarity": cos_sims,
            "Source Word": verify_results["en_strs"],
            "Target Word": verify_results["fr_strs"],
            "Predicted Word": verify_results["top_pred_strs"],
        }
    )

    hover_template = (
        "Rank: %{x}<br>"
        "Cosine Similarity: %{y}<br>"
        "Source Word: %{customdata[0]}<br>"
        "Target Word: %{customdata[1]}<br>"
        "Predicted Word: %{customdata[2]}"
    )

    fig = px.line(
        df,
        x="Rank",
        y="Cosine Similarity",
        title="Cosine Similarity Trend Across Ranks",
        labels={"Cosine Similarity": "Cosine Similarity", "Rank": "Rank"},
        markers=True,
        template="plotly_white",
        custom_data=["Source Word", "Target Word", "Predicted Word"],
    )

    fig.add_scatter(
        x=ranks,
        y=pd.Series(cos_sims).rolling(window=20).mean(),
        mode="lines",
        name="Moving Average",
        line=dict(color="orange", width=2),
    )

    fig.update_traces(
        marker=dict(size=8, color="skyblue", symbol="circle"),
        hovertemplate=hover_template,
    )
    fig.update_layout(
        title_text="Cosine Similarity Between Predicted and Target "
        "Embeddings Across Ranks",
        title_y=0.98,
        legend_orientation="h",
        legend_y=1.2,
    )

    return fig


def test_cosine_similarity_difference(
    verify_results: Dict[str, t.Tensor], n: int = 25
) -> None:
    """Tests for difference in cosine similarity between first and last n entries

    This function performs a two-sample t-test on the first n and last n cosine
    similarity values in a verify_results dictionary to check for a significant
    difference. Results are printed to the console.

    Args:
        verify_results: A dictionary containing the key "cos_sims" with a tensor of
        cosine similarity values as its value.
        n: The number of entries from the start and end to compare. Default is 25.
    """
    cos_sims = verify_results["cos_sims"]
    first_n_cos_sims = cos_sims[:n].cpu()
    last_n_cos_sims = cos_sims[-n:].cpu()

    # Perform a two-sample t-test to check if there's a significant difference
    t_stat, p_value = stats.ttest_ind(first_n_cos_sims, last_n_cos_sims)
    print(f"t-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        print(
            f"There is a significant difference between the first {n} and last {n} "
            "cosine similarity values."
        )
    else:
        print(
            f"There is no significant difference between the first {n} and last {n} "
            "cosine similarity values."
        )


def prepare_verify_analysis(
    model: tl.HookedTransformer, all_word_pairs: List[List[str]], random_seed: int = 1
) -> VerifyWordPairAnalysisResult:
    """Prepares verify analysis by preparing embeddings and calculating distances.

    Args:
        model: The transformer model used for tokenization and generating embeddings.
        all_word_pairs: A collection of word pairs to be analyzed.
        random_seed: An integer used to seed the random number generator.

    Returns:
        A VerifyWordPairAnalysisResult object containing the analysis outcomes.
    """
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    random.seed(random_seed)
    random.shuffle(all_word_pairs)
    selected_index = random.randint(0, len(all_word_pairs) - 1)
    selected_pair = all_word_pairs.pop(selected_index)
    src_word, tgt_word = selected_pair

    # Tokenize and embed the selected word pair
    src_tok = model.tokenizer(
        src_word, return_tensors="pt", add_special_tokens=False
    ).data["input_ids"]
    tgt_tok = model.tokenizer(
        tgt_word, return_tensors="pt", add_special_tokens=False
    ).data["input_ids"]
    src_embed = model.embed.W_E[src_tok].detach().clone().squeeze()
    tgt_embed = model.embed.W_E[tgt_tok].detach().clone().squeeze()

    # Tokenize and embed all other word pairs
    src_other_toks, tgt_other_toks, _, _ = tokenize_word_pairs(model, all_word_pairs)
    src_other_embeds = model.embed.W_E[src_other_toks].detach().clone().squeeze()
    tgt_other_embeds = model.embed.W_E[tgt_other_toks].detach().clone().squeeze()
    other_embeds = t.cat((src_other_embeds, tgt_other_embeds), dim=0)

    # Calculate cosine similarities and Euclidean distances
    src_cos_sims = t.cosine_similarity(src_embed, other_embeds, dim=-1)
    tgt_cos_sims = t.cosine_similarity(tgt_embed, other_embeds, dim=-1)
    src_euc_dists = t.pairwise_distance(src_embed.unsqueeze(0), other_embeds, p=2)
    tgt_euc_dists = t.pairwise_distance(tgt_embed.unsqueeze(0), other_embeds, p=2)

    src_other_data = OtherWords(
        words=all_word_pairs, toks=src_other_toks, embeds=other_embeds
    )
    tgt_other_data = OtherWords(
        words=all_word_pairs, toks=tgt_other_toks, embeds=other_embeds
    )

    src_data = SelectedWord(
        src_word, src_tok, src_embed, src_cos_sims, src_euc_dists, src_other_data
    )
    tgt_data = SelectedWord(
        tgt_word, tgt_tok, tgt_embed, tgt_cos_sims, tgt_euc_dists, tgt_other_data
    )

    return VerifyWordPairAnalysis(
        src=src_data,
        tgt=tgt_data,
    )
