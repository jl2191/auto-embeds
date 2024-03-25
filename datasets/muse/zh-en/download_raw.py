# %%
import requests

from auto_embeds.utils.misc import repo_path_to_abs_path

urls = {
    "https://dl.fbaipublicfiles.com/arrival/dictionaries/zh-en.0-5000.txt": "muse-zh-en-train.txt",
    "https://dl.fbaipublicfiles.com/arrival/dictionaries/zh-en.5000-6500.txt": "muse-zh-en-test.txt",
}

for url, file_name in urls.items():
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code == 200:
        file_path = repo_path_to_abs_path(f"datasets/muse/zh-en/1_raw/{file_name}")
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"{file_name} download successful.")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")
