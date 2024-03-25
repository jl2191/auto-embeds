# %%

import json

from auto_embeds.utils.misc import repo_path_to_abs_path

train_raw_file = repo_path_to_abs_path("datasets/muse/zh-en/1_raw/muse-zh-en-train.txt")
test_raw_file = repo_path_to_abs_path("datasets/muse/zh-en/1_raw/muse-zh-en-test.txt")
train_extracted_save_file = repo_path_to_abs_path(
    "datasets/muse/zh-en/2_extracted/muse-zh-en-train.json"
)
test_extracted_save_file = repo_path_to_abs_path(
    "datasets/muse/zh-en/2_extracted/muse-zh-en-test.json"
)


# %%
def extract_and_save_json(raw_file_path, save_file_path):
    with open(raw_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        word_pairs = [
            [line.strip().split(" ")[0], " ".join(line.strip().split(" ")[1:])]
            for line in lines
        ]  # The initial space delineates the key and value in each line.
        with open(save_file_path, "w", encoding="utf-8") as json_file:
            json.dump(word_pairs, json_file, ensure_ascii=False, indent=4)


extract_and_save_json(train_raw_file, train_extracted_save_file)
extract_and_save_json(test_raw_file, test_extracted_save_file)
