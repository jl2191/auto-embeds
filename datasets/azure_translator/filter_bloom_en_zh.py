# %%
import json

import transformer_lens as tl

from auto_embeds.embed_utils import (
    filter_word_pairs,
)
from auto_embeds.utils.misc import repo_path_to_abs_path

model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")

device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out

with open(
    repo_path_to_abs_path("datasets/cc-cedict/cc-cedict-zh-en-parsed.json"),
    "r",
    encoding="utf-8",
) as file:
    word_pairs = json.load(file)
print(f"Loaded {len(word_pairs)} entries from the dictionary.")
# all_en_fr_pairs =
# for word_pair in word_pairs:
    # print(word_pair)

# %%
# random.seed(1)
# all_en_fr_pairs.sort(key=lambda pair: pair[0])
# word_pairs.sort(key=lambda pair: pair[1])
# random.shuffle(word_pairs)
# split_index = int(len(word_pairs) * 0.95)
# train_en_fr_pairs = word_pairs[:split_index]
# test_en_fr_pairs = word_pairs[split_index:]

filtered_word_pairs = filter_word_pairs(
    model,
    word_pairs,
    discard_if_same=True,
    min_length=1,
    # capture_diff_case=True,
    capture_space=True,
    capture_no_space=True,
    print_pairs=True,
    print_number=True,
    # max_token_id=200_000,
    most_common_english=True,
    most_common_french=True,
    acceptable_overlap=0.8,
)

filtered_word_pairs.sort(key=lambda pair: pair[1])
# %%
file_name = repo_path_to_abs_path("datasets/azure_translator/bloom-zh-en-zh-only.json")
chinese_words = {pair[0].strip().lower() for pair in filtered_word_pairs}
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(list(chinese_words), f, ensure_ascii=False, indent=4)

print(f"Saved all unique Chinese words to the current directory as {file_name}")
