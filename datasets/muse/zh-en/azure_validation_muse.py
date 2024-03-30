# %%

import json
import os
import time  # Importing the time module for delays

import requests

from auto_embeds.utils.misc import repo_path_to_abs_path


def get_subscription_key():
    # Try to get the subscription key from an environment variable
    key = os.getenv("AZURE_TRANSLATOR_KEY")
    if not key:
        # If the key is not found in the environment variables, prompt the user for it
        key = input("Enter your Azure Translator subscription key: ")
    return key


subscription_key = get_subscription_key()
endpoint = "https://api.cognitive.microsofttranslator.com"
path = "/dictionary/lookup?api-version=3.0&from=zh-Hant&to=en"

headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-Type": "application/json",
}


def lookup_translations(words):
    constructed_url = endpoint + path
    body = [{"Text": word} for word in words]
    response = requests.post(constructed_url, headers=headers, json=body).json()
    print(
        json.dumps(response, indent=4, ensure_ascii=False)
    )  # Debugging: Print the response
    time.sleep(5)  # Adding a 5 second delay between API calls for debugging purposes
    return response


def process_file(input_files, output_file):

    all_word_pairs = []

    for input_file in input_files:
        with open(input_file, "r") as file:
            word_pairs = json.load(file)
            all_word_pairs.extend(word_pairs)

    # Extract Chinese words only from the filtered word pairs
    chinese_words = {pair[0].strip() for pair in all_word_pairs}

    print(f"Number of unique Chinese words: {len(chinese_words)}")

    translations = []
    total_words = len(chinese_words)
    print(f"Total words to process: {total_words}")

    # Batch process words, there's a limit of 10 at a time
    for i in range(0, total_words, 10):
        batch = list(chinese_words)[i : i + 10]
        batch_translations = lookup_translations(batch)
        translations.extend(batch_translations)
        processed = min(i + 10, total_words)
        print(f"Processed {processed}/{total_words} words...")

    with open(output_file, "w") as file:
        json.dump(translations, file, ensure_ascii=False, indent=4)


input_files = [
    repo_path_to_abs_path("datasets/muse/zh-en/3_filtered/muse-zh-en-train.json"),
    repo_path_to_abs_path("datasets/muse/zh-en/3_filtered/muse-zh-en-test.json"),
]
output_file = repo_path_to_abs_path(
    "datasets/muse/zh-en/4_azure_validation/muse-zh-en-azure-val.json"
)

process_file(input_files, output_file)

# %%
