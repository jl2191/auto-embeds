# %%
import torch as t
import json
import transformer_lens as tl
import random
from auto_embeds.utils.misc import repo_path_to_abs_path

# %%
model = tl.HookedTransformer.from_pretrained("bloom-560m")

vocab_size = model.cfg.d_vocab_out  # Get the size of the vocabulary

# Generate random indices
num_pairs = 20000
random_indices = [[random.randint(0, vocab_size - 1)] for _ in range(num_pairs * 2)]

random_words = model.to_string(random_indices)
random_word_pairs = [
    (random_words[i].strip(), random_words[i + 1].strip())
    for i in range(0, len(random_indices), 2)
]

print(random_word_pairs)
# %%
# Generate the save path using the repo_path_to_abs_path function
save_path = repo_path_to_abs_path("datasets/random/random_word_pairs.json")

# Save the random word pairs as JSON
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(random_word_pairs, f, ensure_ascii=False, indent=4)

print(f"Saved {len(random_word_pairs)} random word pairs to {save_path}")
