# %%
import nltk
import json
import inflect
from nltk.corpus import wordnet as wn
from auto_embeds.utils.misc import repo_path_to_abs_path

# Ensure that WordNet and inflect are downloaded
nltk.download("wordnet")


def generate_singular_plural_pairs():
    """Generate singular and plural pairs of nouns, excluding numeric entries."""
    p = inflect.engine()
    nouns = list(wn.all_lemma_names(pos="n"))
    pairs = []

    for noun in nouns:
        if noun.isnumeric():  # Skip numeric entries
            continue
        plural = p.plural_noun(noun)
        if plural:
            pairs.append([noun, plural])
    return pairs


def save_to_json(pairs, filename="datasets/singular-plural/singular_plural_pairs.json"):
    """Save the singular-plural pairs to a JSON file as a list."""
    save_path = repo_path_to_abs_path(filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    pairs = generate_singular_plural_pairs()
    save_to_json(pairs)
    print(f"Generated {len(pairs)} singular-plural pairs.")
