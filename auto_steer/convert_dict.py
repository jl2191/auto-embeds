# %%
import json

from auto_steer.utils.misc import repo_path_to_abs_path

# %%
# Load the original JSON data
data = []
with open(
    repo_path_to_abs_path("datasets/kaikki-french-dictionary.json"),
    "r",
    encoding="utf-8",
) as file:
    for line in file:
        data.append(json.loads(line))
print(f"Loaded {len(data)} entries from the dictionary.")

# Initialize a list to hold the translation pairs
translation_pairs = []

# Extract translation pairs
for entry in data:
    french_word = entry.get("word", "")
    # Check if the entry has senses and glosses for extracting English translations
    if "senses" in entry and entry["senses"]:
        for sense in entry["senses"]:
            if "glosses" in sense and sense["glosses"]:
                # Extract all English translations from the glosses
                english_translations = [
                    gloss
                    for gloss in sense["glosses"]
                    if " " not in gloss and "-" not in gloss
                ]
                # Create a pair for each English translation
                for english_word in english_translations:
                    translation_pairs.append(
                        {"French": french_word, "English": english_word}
                    )
                # Stop after adding translations from the first sense with glosses
                break
# Write the new JSON structure to a file
with open(
    "kaikki-french-dictionary-single-word-pairs-no-hyphen.json", "w", encoding="utf-8"
) as outfile:
    json.dump(translation_pairs, outfile, ensure_ascii=False, indent=4)
