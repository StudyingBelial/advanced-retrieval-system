from datasets import load_dataset
import re

def get_processed_data():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split = "test")
    title_pattern = r'(=+) \s*(.*?) \s*\1'
    filtered_data = []
    current_chunk = {}
    current_paragraph = []
    for index, data in enumerate(ds):
        text = data["text"].strip()
        if text == "":
            continue
        elif text[0] == '=':
            match = re.match(title_pattern, text)  # Match from the start
            if match and match.group(0) == text:  # Ensure the entire text matches
                title = match.group(2).strip()  # Extract the title
                if title:  # Only print non-empty titles
                    current_chunk["text"] = current_paragraph
                    filtered_data.append(current_chunk)
                    current_paragraph = []
                    current_chunk = {}
                    current_chunk["title"] = title.replace(" @-@ ", "-")
        else:
            current_paragraph.append(text.replace(" @-@ ", "-").replace(" \'s", "'s"))
    filtered_data.pop(0)
    unspaced_data = [{key : " ".join(data)} for key, data in filtered_data]
    return unspaced_data