# %%
import json
import os

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
path = "/dictionary/lookup?api-version=3.0&from=zh-Hans&to=en"

headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-Type": "application/json",
}


def lookup_translations(words):
    constructed_url = endpoint + path
    body = [{"Text": word} for word in words]
    response = requests.post(constructed_url, headers=headers, json=body).json()
    # if words
    # print(
    #     json.dumps(response, indent=4, ensure_ascii=False)
    # )  # Debugging: Print the response
    return response


def process_file(input_file, output_file):
    with open(input_file, "r") as file:
        words = json.load(file)

    translations = []
    total_words = len(words)
    print(f"Total words to process: {total_words}")

    # Process words in batches of 10 as this is the max words that can be sent at a time
    for i in range(0, total_words, 10):
        batch = words[i : i + 10]
        batch_translations = lookup_translations(batch)
        translations.extend(batch_translations)
        processed = min(i + 10, total_words)
        print(f"Processed {processed}/{total_words} words...")

    with open(output_file, "w") as file:
        json.dump(translations, file, ensure_ascii=False, indent=4)


input_file = repo_path_to_abs_path("datasets/azure_translator/bloom-zh-en-zh-only.json")
output_file = repo_path_to_abs_path(
    "datasets/azure_translator/bloom-zh-en-all-translations.json"
)

process_file(input_file, output_file)

# %%
