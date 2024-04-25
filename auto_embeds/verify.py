import random
from typing import Any, Dict, List, Tuple, Union

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch as t
import torch.nn as nn
import transformer_lens as tl
from plotly.graph_objects import Figure
from rich.table import Table
from scipy import stats
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import (
    ExtendedWordData,
    VerifyWordPairAnalysis,
    WordCategory,
    WordData,
    tokenize_word_pairs,
)
from auto_embeds.utils.misc import calculate_gradient_color, default_device


def verify_transform(
    model: tl.HookedTransformer,
    transformation: nn.Module,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    unembed_module: nn.Module,
) -> Dict[str, Any]:
    """Evaluates the transformation's effectiveness.

    Evaluates the transformation's effectiveness in translating source language tokens
    to target language tokens by examining the relationship between accuracy and the
    cosine similarity of embeddings. Returns a dictionary containing cosine similarity,
    Euclidean distance, and strings for source, target, and predicted tokens.

    Args:
        model: A transformer model with hooked embeddings and a tokenizer.
        transformation: The transformation module applied to source language embeddings.
        test_loader: DataLoader for the test dataset with source and
                     target language embeddings tuples.
        unembed_module: The module used for unembedding.

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
            en_logits = unembed_module(en_embeds)
            en_strs: List[str] = model.tokenizer.batch_decode(
                en_logits.squeeze().argmax(dim=-1)
            )
            fr_logits = unembed_module(fr_embeds)
            fr_strs: List[str] = model.tokenizer.batch_decode(
                fr_logits.squeeze().argmax(dim=-1)
            )
            # transform it using our learned rotation
            with t.no_grad():
                top_pred_embeds = transformation(en_embeds)
            # get the top predicted tokens
            top_pred_logits = unembed_module(top_pred_embeds)
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
    """Returns a rich table.

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
    all_word_pairs: List[List[str]],
    device: Union[str, t.device] = default_device,
) -> Dict[str, Union[str, List[str]]]:
    """Calculates the percentage of target tokens in top closest.

    Calculates the percentage of instances where the target language token is within
    the top 1 and top 5 closest tokens in terms of cosine similarity.

    Args:
        model: The model used for embedding and token decoding.
        all_word_pairs: A list of tuples containing source and target word pairs.
        device: The device on which to allocate tensors. If None, defaults to
            default_device

    Returns:
        A dictionary containing a summary of the results and detailed results for each
        source token. Keys are ["summary"] and ["details"].
    """
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    correct_count_top_5 = 0
    correct_count_top_1 = 0
    total_count = len(all_word_pairs)

    details = []

    src_toks, tgt_toks, _, _ = tokenize_word_pairs(model, all_word_pairs)

    all_toks = t.cat([src_toks, tgt_toks], dim=0)
    all_embeds = model.embed.W_E[all_toks].detach().clone()

    for i, (src_tok, correct_tgt_tok) in enumerate(zip(src_toks, tgt_toks)):
        # Embed the source token
        src_embed = model.embed.W_E[src_tok].detach().clone().squeeze(0)
        # shape [d_model]

        # Exclude the current source token from other_toks and other_embeds to avoid
        # self-matching
        valid_indices = (all_toks != src_tok).squeeze(-1)
        # the mask valid_indices should be of shape [batch]

        valid_other_toks = all_toks[valid_indices]
        valid_other_embeds = all_embeds[valid_indices].squeeze(1)
        # should now both be of shape [batch]

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
    word_category_data: WordCategory,
    sort_by: str = "cos_sim",
    display_limit: int = 50,
    top_k: int = 200,
    exclude_identical: bool = False,
    device: Union[str, t.device] = default_device,
) -> Table:
    """Generates a table of top word pairs.

    Highlights the relationship between a selected word and other tokens based on
    either cosine similarity or Euclidean distance. It displays a limited number of
    top entries as specified by the user. Optionally excludes identical tokens to
    the selected word to ensure more meaningful comparisons.

    Args:
        model: The model used for token decoding.
        selected_word_data: Data for the selected word including other words to compare.
        sort_by: Criterion for sorting the tokens ('cos_sim' or 'euc_dist').
        display_limit: Number of entries to display in the table.
        top_k: The number of top entries to consider for any given metric.
        exclude_identical: If True, identical tokens to the selected word are excluded.

    Returns:
        A rich Table object containing the ranked tokens, their cosine similarity,
        and Euclidean distance to the selected word, excluding identical tokens if
        specified.
    """
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    # Preprocess tokens to exclude identical ones if requested
    if exclude_identical:
        identical_token_index = t.where(
            word_category_data.other.toks == word_category_data.selected.toks
        )[0]
        mask = t.ones(
            len(word_category_data.other.toks),
            dtype=t.bool,
            device=device,
        )
        mask[identical_token_index] = False
    else:
        mask = t.ones(
            len(word_category_data.other.toks),
            dtype=t.bool,
            device=device,
        )

    if sort_by == "cos_sim":
        title_sort_by = "Cosine Similarity"
        sorted_indices = word_category_data.other.cos_sims.argsort(descending=True)[
            :top_k
        ]
    elif sort_by == "euc_dist":
        title_sort_by = "Euclidean Distance"
        sorted_indices = word_category_data.other.euc_dists.argsort()[:top_k]
    else:
        raise ValueError("Supported sort functions are 'cos_sim' and 'euc_dist'")

    table = Table(
        show_header=True,
        title=f"Closest tokens to [plum3 on grey23]"
        "{word_category_data.selected.words[0]}"
        f"[/plum3 on grey23] sorted by {title_sort_by}",
    )
    table.add_column("Rank")
    table.add_column("Token")
    table.add_column("Cos Sim")
    table.add_column("Euc Dist")

    # Apply mask to exclude identical tokens if needed
    other_toks = word_category_data.other.toks[sorted_indices][mask[sorted_indices]]
    cos_sims = word_category_data.other.cos_sims[sorted_indices][mask[sorted_indices]]
    euc_dists = word_category_data.other.euc_dists[sorted_indices][mask[sorted_indices]]

    # Determine the range of cosine similarities and euclidean distances for gradient
    # calculation
    cos_sim_values = cos_sims.tolist()
    euc_dist_values = euc_dists.tolist()
    cos_sim_min, cos_sim_max = min(cos_sim_values), max(cos_sim_values)
    euc_dist_min, euc_dist_max = min(euc_dist_values), max(euc_dist_values)

    display_start = display_limit // 2
    for rank, (token, cos_sim, euc_dist) in enumerate(
        zip(other_toks, cos_sims, euc_dists), 1
    ):
        if rank == display_start + 1:
            table.add_row("...", "...", "...", "...")
        elif rank > display_start and rank <= len(other_toks) - display_start:
            continue
        else:
            word = model.tokenizer.decode(token)
            cos_sim = cos_sim.item()
            euc_dist = euc_dist.item()

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
    """Plots the trend of cosine similarity across ranks.

    Plots the trend of cosine similarity across ranks with the ability to toggle
    between a line of best fit and a moving average trend line via the legend.
    Rolling standard deviation for the moving average is shown when the moving average
    is toggled on. Dots are colored green when the predicted word matches the target
    word, otherwise red.

    Args:
        verify_results: A dictionary containing the cosine similarities, source words,
            target words, and predicted words.

    Returns:
        A Plotly Figure object representing the trend of cosine similarity across ranks
        with interactive options for viewing the line of best fit and moving average.
    """
    cos_sims = verify_results["cos_sims"].tolist()
    ranks = list(range(len(cos_sims)))
    target_words = verify_results["fr_strs"]
    predicted_words = verify_results["top_pred_strs"]

    # Prepare DataFrame for plotting
    df = pd.DataFrame(
        {
            "Rank": ranks,
            "Cosine Similarity": cos_sims,
            "Source Word": verify_results["en_strs"],
            "Target Word": target_words,
            "Predicted Word": predicted_words,
        }
    )

    # Define hover template for detailed information on hover
    hover_template = (
        "Rank: %{x}<br>"
        "Cosine Similarity: %{y}<br>"
        "Source Word: %{customdata[0]}<br>"
        "Target Word: %{customdata[1]}<br>"
        "Predicted Word: %{customdata[2]}"
    )

    # Determine dot colors based on word match
    dot_colors = [
        "lightgreen" if target == pred else "lightcoral"
        for target, pred in zip(target_words, predicted_words)
    ]

    # Create the main line plot with conditional coloring
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

    # Update dots with conditional colors
    fig.update_traces(
        marker=dict(size=8, color=dot_colors, symbol="circle"),
        hovertemplate=hover_template,
    )

    # Calculate and add line of best fit
    slope, intercept = np.polyfit(ranks, cos_sims, 1)
    line_of_best_fit = [slope * x + intercept for x in ranks]
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=line_of_best_fit,
            mode="lines",
            name="Line of Best Fit",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    # Calculate and add moving average
    moving_avg = pd.Series(cos_sims).rolling(window=20).mean()
    moving_std = pd.Series(cos_sims).rolling(window=20).std()
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=moving_avg,
            mode="lines",
            name="Moving Average",
            line=dict(color="orange", width=2),
        )
    )

    # Add rolling std dev for the moving average, shown with the moving average
    fig.add_trace(
        go.Scatter(
            x=ranks + ranks[::-1],  # x, then x reversed
            y=(moving_avg + moving_std).tolist()
            + (moving_avg - moving_std).tolist()[::-1],
            fill="toself",
            fillcolor="rgba(255,165,0,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Rolling Std Dev",
            visible="legendonly",  # Shown with moving average
        )
    )

    # Update layout for a cleaner look
    fig.update_layout(
        title_text="Cosine Similarity Between Predicted and "
        "Target Embeddings Across Ranks",
        title_y=0.98,
        legend_orientation="h",
        legend_y=1.2,
    )

    return fig


def test_cos_sim_difference(
    verify_results: Dict[str, t.Tensor], n: int = 25
) -> Dict[str, Union[float, bool, int]]:
    """Tests for difference in cosine similarity.

    Tests for difference in cosine similarity between first and last n entries.
    This function performs a two-sample t-test on the first n and last n cosine
    similarity values in a verify_results dictionary to check for a significant
    difference. The results are returned in a dictionary.

    Args:
        verify_results: A dictionary containing the key "cos_sims" with a tensor of
        cosine similarity values as its value.
        n: The number of entries from the start and end to compare. Default is 25.

    Returns:
        A dictionary with keys "t-statistic", "P-value", "significant_difference"
        and "n" where "significant_difference" is a boolean indicating whether the
        difference is statistically significant, and "n" is the number of entries
        compared from each end.
    """
    cos_sims = verify_results["cos_sims"]
    first_n_cos_sims = cos_sims[:n].cpu()
    last_n_cos_sims = cos_sims[-n:].cpu()

    # Perform a two-sample t-test to check if there's a significant difference
    t_stat, p_value = stats.ttest_ind(first_n_cos_sims, last_n_cos_sims)
    significant_difference = p_value < 0.05

    return {
        "t-statistic": t_stat,
        "p-value": p_value,
        "significant_difference": significant_difference,
        "n": n,
    }


def prepare_verify_analysis(
    model: tl.HookedTransformer,
    embed_module: nn.Module,
    all_word_pairs: List[List[str]],
    seed: int = 1,
    device: Union[str, t.device] = default_device,
    keep_other_pair: bool = False,
) -> VerifyWordPairAnalysis:
    """Prepares verify analysis.

    Prepares verify analysis by preparing embeddings and calculating distances.
    This function calculates and stores the top indices for cosine similarities and
    Euclidean distances within the WordCategory data structure, removing the need to
    pass these indices separately to other functions. If keep_other_pair is True,
    src_other_toks will include the tgt_tok and vice versa.

    Args:
        model: The transformer model used for tokenization and generating embeddings.
        embed_module: The module used for embedding.
        all_word_pairs: A collection of word pairs to be analyzed.
        seed: An integer used to seed the random number generator.
        device: The device on which to allocate tensors. If None, defaults to
            default_device.
        keep_other_pair: If True, includes the target token in src_other_toks and
            the source token in tgt_other_toks.

    Returns:
        A VerifyWordPairAnalysis object containing the analysis outcomes.
    """

    all_word_pairs_copy = all_word_pairs.copy()
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")
    random.seed(seed)
    random.shuffle(all_word_pairs_copy)
    selected_index = random.randint(0, len(all_word_pairs_copy) - 1)
    selected_pair = all_word_pairs_copy.pop(selected_index)
    src_word, tgt_word = selected_pair

    # for our selected src and tgt word
    ## tokenize
    src_tok = (
        model.tokenizer(src_word, return_tensors="pt", add_special_tokens=False)
        .data["input_ids"]
        .to(device)
    )
    tgt_tok = (
        model.tokenizer(tgt_word, return_tensors="pt", add_special_tokens=False)
        .data["input_ids"]
        .to(device)
    )
    ## embed
    src_embed = embed_module(src_tok).detach().clone().squeeze()
    tgt_embed = embed_module(tgt_tok).detach().clone().squeeze()

    src_other_words = [word_pair[0] for word_pair in all_word_pairs_copy]
    tgt_other_words = [word_pair[1] for word_pair in all_word_pairs_copy]

    if keep_other_pair:
        src_other_words.append(tgt_word)
        tgt_other_words.append(src_word)
    # tokenize
    src_other_toks = (
        model.tokenizer(src_other_words, return_tensors="pt", add_special_tokens=False)
        .data["input_ids"]
        .to(device)
    )
    tgt_other_toks = (
        model.tokenizer(tgt_other_words, return_tensors="pt", add_special_tokens=False)
        .data["input_ids"]
        .to(device)
    )
    ## embed
    src_other_embeds = embed_module(src_other_toks).detach().clone().squeeze(1)
    tgt_other_embeds = embed_module(tgt_other_toks).detach().clone().squeeze(1)
    # both should have shape [batch, d_model]

    # calculate cosine similarities and euclidean distances
    src_cos_sims = t.cosine_similarity(src_embed, src_other_embeds, dim=-1)
    tgt_cos_sims = t.cosine_similarity(tgt_embed, tgt_other_embeds, dim=-1)
    src_euc_dists = t.pairwise_distance(src_embed.unsqueeze(0), src_other_embeds, p=2)
    tgt_euc_dists = t.pairwise_distance(tgt_embed.unsqueeze(0), tgt_other_embeds, p=2)

    src_other_data = ExtendedWordData(
        words=src_other_words,
        toks=src_other_toks,
        embeds=src_other_embeds,
        cos_sims=src_cos_sims,
        euc_dists=src_euc_dists,
    )
    tgt_other_data = ExtendedWordData(
        words=tgt_other_words,
        toks=tgt_other_toks,
        embeds=tgt_other_embeds,
        cos_sims=tgt_cos_sims,
        euc_dists=tgt_euc_dists,
    )

    src_data = WordData(words=[src_word], toks=src_tok, embeds=src_embed)
    tgt_data = WordData(words=[tgt_word], toks=tgt_tok, embeds=tgt_embed)

    src_category = WordCategory(selected=src_data, other=src_other_data)
    tgt_category = WordCategory(selected=tgt_data, other=tgt_other_data)

    return VerifyWordPairAnalysis(
        src=src_category,
        tgt=tgt_category,
    )


def prepare_verify_datasets(
    verify_learning,
    batch_sizes=(64, 256),
    top_k=200,
    seed=None,
    top_k_selection_method="src_and_src",
):
    """Prepares training and testing datasets.

    Prepares training and testing datasets from embeddings, selecting top-k
    embeddings based on cosine similarity and allows for deterministic shuffling
    with a specified seed. The selection can be based on the cosine similarities
    between source-source embeddings, target-target embeddings, or selecting the
    entire word pair based on top cosine similarity from a randomly chosen
    source or target embedding.

    Args:
        verify_analysis: An object with source and target embeddings generated from
            prepare_verify_analysis.
        batch_sizes: A tuple with the batch sizes for training and testing datasets.
        top_k: Number of top embeddings to select based on cosine similarity.
        seed: Optional; A seed for deterministic shuffling using a torch generator.
        top_k_selection_method: The procedure to use to select the embeddings to verify
            with. src_and_src selects based on the cosine similarities of the source
            embeddings, tgt_and_tgt for target embeddings, top_src and top_tgt for
            selecting the entire word pair based on top cosine similarity from a
            randomly chosen source or target embedding, respectively.

    Returns:
        A tuple with DataLoader objects for the training and testing datasets.
    """
    # Create a generator for deterministic shuffling if seed is provided
    generator = t.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # This function initializes the random seeds for worker processes in DataLoader
    def seed_worker(worker_seed):
        worker_seed = t.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create test_embeds tensor from top 200 indices
    # other_embeds is of shape [batch, d_model] at this point
    # and src_top_200_cos_sim_indices is of shape [batch] and we want to select all the
    # the embeddings with the top 200 cos sims
    src_embed = verify_learning.src.selected.embeds
    src_embeds = verify_learning.src.other.embeds
    tgt_embed = verify_learning.tgt.selected.embeds
    tgt_embeds = verify_learning.tgt.other.embeds

    if top_k_selection_method == "src_and_src":
        cos_sims = t.cosine_similarity(src_embed, src_embeds, dim=-1)
        top_k_cos_sims = t.topk(cos_sims, top_k, largest=True)
        test_indices = top_k_cos_sims.indices
    elif top_k_selection_method == "tgt_and_tgt":
        cos_sims = t.cosine_similarity(tgt_embed, tgt_embeds, dim=-1)
        top_k_cos_sims = t.topk(cos_sims, top_k, largest=True)
        test_indices = top_k_cos_sims.indices
    elif top_k_selection_method in ["top_src", "top_tgt"]:
        # Randomly select an embedding from src or tgt based on the selection
        random_embed = src_embed if top_k_selection_method == "top_src" else tgt_embed
        # get the top cosine similarities with both the src and tgt embeds
        # as both the src and tgt embeds line up at this stage, the same index should
        # give the corresponding word pair in the other tensor. as such, to get our
        # indices, we first get the indices of our random_embed with all our other
        # embeds.
        random_embed_w_src_embeds_cos_sims = t.cosine_similarity(
            random_embed, src_embeds, dim=-1
        )
        random_embed_w_tgt_embeds_cos_sims = t.cosine_similarity(
            random_embed, tgt_embeds, dim=-1
        )
        random_embed_w_all_embeds_cos_sims = t.cat(
            (random_embed_w_src_embeds_cos_sims, random_embed_w_tgt_embeds_cos_sims)
        )
        # get the indices of the top k cos sims for both src and tgt
        _, indices = t.topk(
            random_embed_w_all_embeds_cos_sims,
            random_embed_w_all_embeds_cos_sims.shape[0],
        )
        # turn it into a list to get the correct indices for tgt embeds as they are
        # offset by the src embeds
        indices = indices.tolist()
        indices = [
            index - src_embeds.shape[0] if index >= src_embeds.shape[0] else index
            for index in indices
        ]
        test_indices = t.tensor(indices)
        test_indices = t.unique_consecutive(test_indices)[:top_k]
    else:
        raise ValueError(
            "Invalid top_k_selection_method value. Accepted values are 'src_and_src', \
                'tgt_and_tgt', 'top_src', 'top_tgt'."
        )

    # Index into src and tgt embeds with these top-k indices to get test embeddings
    src_embeds_with_top_k_cos_sims = src_embeds[test_indices]
    tgt_embeds_with_top_k_cos_sims = tgt_embeds[test_indices]

    # Unsqueeze to add an extra dimension for pos
    src_test_embeds = src_embeds_with_top_k_cos_sims.unsqueeze(1)
    tgt_test_embeds = tgt_embeds_with_top_k_cos_sims.unsqueeze(1)
    # these should now be [batch, pos, d_model]

    print(f"source test embeds shape: {src_test_embeds.shape}")
    print(f"target test embeds shape: {tgt_test_embeds.shape}")

    # For train embeddings, we just need to remove the indices we used for the test
    mask = t.ones(src_embeds.shape[0], dtype=t.bool, device=src_embeds.device)
    mask[test_indices] = False

    src_train_embeds = src_embeds[mask].unsqueeze(1)
    tgt_train_embeds = tgt_embeds[mask].unsqueeze(1)
    # these should now be [batch, pos, d_model]

    print(f"source train embeds shape: {src_train_embeds.shape}")
    print(f"target train embeds shape: {tgt_train_embeds.shape}")

    # Prepare DataLoader objects for training and testing datasets
    train_dataset = TensorDataset(src_train_embeds, tgt_train_embeds)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_sizes[0],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_dataset = TensorDataset(src_test_embeds, tgt_test_embeds)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_sizes[1],
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, test_loader
