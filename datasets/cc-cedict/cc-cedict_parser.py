# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parser class for cedict Chinese-English dictionary files
"""

import os
import os.path
import io
import logging
import pickle

from os import path
from auto_embeds.utils.misc import repo_path_to_abs_path

CEDICT_PATH = repo_path_to_abs_path("datasets/cc-cedict/cedict_ts.u8")
DATA_PATH = repo_path_to_abs_path("datasets/cc-cedict/dump.dat")


class CedictParser:
    """Parser class. Reads a cedict file and return a list of
    CedictEntry instances with each line processed.
    """

    filters = ["_filter_comments", "_filter_new_lines", "_filter_empty_entries"]

    def __init__(self, lines=None, file_path=None, lines_count=None):
        self.lines = lines or []
        self.lines_count = lines_count

        if not file_path and os.path.isfile(DATA_PATH):
            self.lines = pickle.load(open(DATA_PATH, "rb"))
        elif file_path:
            self.read_file(file_path)
        else:
            self.read_file(file_path=CEDICT_PATH)

    def read_file(self, file_path):
        """Import the cedict file sanitizing each entry"""
        with io.open(file_path, "r", encoding="utf-8") as file_handler:
            if self.lines_count:
                logging.info("Loaded %s lines of the dictionary", self.lines_count)
            self.lines = file_handler.readlines()
            self._sanitize()
            pickle.dump(self.lines, open(DATA_PATH, "wb"))

    def _sanitize(self):
        from operator import methodcaller

        # f = methodcaller('_filter_comments')
        # f(b) returns b._filter_comments().
        for fun in self.filters:
            caller = methodcaller(fun)
            caller(self)

    def _filter_comments(self):
        """remove lines starting with # or #!"""
        self.lines = [line for line in self.lines if not line.startswith(("#"))]

    def _filter_new_lines(self):
        self.lines = [line.rstrip("\n") for line in self.lines]

    def _filter_empty_entries(self):
        self.lines = [line for line in self.lines if line.strip()]

    def parse(self):
        """Parse Cedict lines and return a list of CedictEntry items"""
        result = []
        for line in self.lines[: self.lines_count]:
            entry = CedictEntry.make(line)
            result.append(entry)
        return result


class CedictEntry:  # pylint: disable=too-few-public-methods
    """A representation of a cedict entry

    Keyword arguments:
    traditional -- entry in traditional hanzi
    simplified -- entry in simplified hanzi
    pinyin -- entry pronunciation with tone numbers
    meanings -- list of different meanings for an entry
    raw_line -- the original full line
    """

    def __init__(self, **kwargs):
        self.traditional = kwargs.get("traditional", "")
        self.simplified = kwargs.get("simplified", "")
        self.pinyin = kwargs.get("pinyin", "")
        self.meanings = kwargs.get("meanings", "")
        self.raw_line = kwargs.get("raw_line", "")

    @classmethod
    def make(cls, line):
        """Generates an entry from a Cedict file line data"""
        hanzis = line.partition("[")[0].split(" ", 1)
        keywords = dict(
            meanings=line.partition("/")[2]
            .replace('"', "'")
            .rstrip("/")
            .strip()
            .split("/"),
            traditional=hanzis[0].strip(" "),
            simplified=hanzis[1].strip(" "),
            # Take the content in between the two brackets
            pinyin=line.partition("[")[2].partition("]")[0],
            raw_line=line,
        )
        return cls(**keywords)

    def __str__(self):
        return "[{}], [{}]".format(self.simplified, self.meanings)

    def __list__(self):
        return [self.simplified, self.meanings]


# %%
parser = CedictParser()
entries = parser.parse()
dict_list = [[entry.simplified, entry.meanings] for entry in entries]

# for e in entries:
#     entry = str(e)
#     dict_list.append(entry)
# print(len(dict_list))

# %%
import json

# with open("cedict_entries.json", "w", encoding="utf-8") as file:
#     json.dump(dict_list, file, ensure_ascii=False, indent=4)

# %%
# with open("cedict_entries.json", "r", encoding="utf-8") as file:
#     dict_list = json.load(file)


# %%
print(dict_list[:30])
print(len(dict_list))
print(type(dict_list[0]))

# %%
new_dict_list = []
for entry in dict_list:
    word = entry[0]
    meanings = entry[1]
    if isinstance(meanings, list):
        for meaning in meanings:
            new_dict_list.append([word, meaning])
    else:
        new_dict_list.append([word, meanings])
dict_list = new_dict_list

# %%
import re

dict_list = [entry for entry in dict_list if "variant" not in entry[1]]
filtered_dict_list = []
for entry in dict_list:
    simplified, meaning = entry
    # Remove text within parentheses
    cleaned_meaning = re.sub(r"\(.*?\)", "", meaning).strip()
    # Keep the word after "to" if it exists
    if "to " in cleaned_meaning:
        cleaned_meaning = cleaned_meaning.split("to ")[-1]
        # print(cleaned_meaning)
    filtered_dict_list.append([simplified, cleaned_meaning])

dict_list = filtered_dict_list

# %%
print(len(dict_list))
dict_list = [
    entry
    for entry in dict_list
    if len(entry[1].split()) == 1 and all(not char.isascii() for char in entry[0])
]

# %%
save_path = repo_path_to_abs_path("datasets/cc-cedict/cc-cedict-zh-en-parsed.json")
with open(save_path, "w", encoding="utf-8") as file:
    json.dump(dict_list, file, ensure_ascii=False, indent=4)
