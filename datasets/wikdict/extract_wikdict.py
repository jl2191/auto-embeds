# %%

import glob
from lxml import etree
import json
import os
from auto_steer.utils.misc import repo_path_to_abs_path

unprocessed_directory = repo_path_to_abs_path("datasets/wikdict/1_raw")
extracted_directory = repo_path_to_abs_path("datasets/wikdict/2_extracted")


def extract_word_pairs(xml_file, src_lang, tgt_lang):
    tree = etree.parse(xml_file)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    entries = tree.xpath("//tei:entry", namespaces=ns)

    word_pairs = []

    for entry in entries:
        src_words = entry.xpath(".//tei:form/tei:orth/text()", namespaces=ns)
        tgt_translations = entry.xpath(
            f'.//tei:sense/tei:cit[@type="trans"]/tei:quote/text()',
            namespaces=ns,
        )

        for src_word in src_words:
            for tgt_word in tgt_translations:
                if (
                    src_word != tgt_word
                    and not src_word.isdigit()
                    and not tgt_word.isdigit()
                ):
                    word_pairs.append([src_word, tgt_word])

    return word_pairs


def process_files(directory):
    for file_path in sorted(glob.glob(f"{directory}/*.tei")):
        filename = os.path.basename(file_path)
        src_lang, tgt_lang = (
            filename.split("-")[0],
            filename.split("-")[1].split(".")[0],
        )
        word_pairs = extract_word_pairs(file_path, src_lang, tgt_lang)

        output_file_path = f"{extracted_directory}/{src_lang}-{tgt_lang}.json"
        with open(output_file_path, "w", encoding="utf-8") as json_file:
            json.dump(word_pairs, json_file, ensure_ascii=False, indent=4)

        print(f"Exported {len(word_pairs)} word pairs to '{output_file_path}'")


process_files(unprocessed_directory)
"""
Exported 267379 word pairs to '/root/auto-steer/datasets/wikdict/2_extracted/eng-fin.json'
Exported 186975 word pairs to '/root/auto-steer/datasets/wikdict/2_extracted/eng-fra.json'
Exported 114113 word pairs to '/root/auto-steer/datasets/wikdict/2_extracted/eng-jpn.json'
Exported 143160 word pairs to '/root/auto-steer/datasets/wikdict/2_extracted/eng-pol.json'
Exported 241820 word pairs to '/root/auto-steer/datasets/wikdict/2_extracted/eng-rus.json'
Exported 76567 word pairs to '/root/auto-steer/datasets/wikdict/2_extracted/eng-tur.json'
"""
# %%
