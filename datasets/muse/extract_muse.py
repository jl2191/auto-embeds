# %%

import glob
from lxml import etree
import json
import os
from auto_embeds.utils.misc import repo_path_to_abs_path

raw_file = repo_path_to_abs_path("datasets/muse/1_raw/en-fr.txt")
extracted_save_file = repo_path_to_abs_path("datasets/muse/2_extracted/en-fr.json")

# %%
def extract_and_save_json(raw_file_path, save_file_path):
    with open(raw_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        word_pairs = [line.strip().split(' ') for line in lines]
        with open(save_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(word_pairs, json_file, ensure_ascii=False, indent=4)

extract_and_save_json(raw_file, extracted_save_file)
