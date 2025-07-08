#!/usr/bin/env python3
"""
upload text2emoji model to hugging face
"""

import os
from huggingface_hub import HfApi, create_repo
from model import Text2EmojiModel, Text2EmojiConfig

def upload_model():
    """upload model to hugging face"""
    
    # create repo
    repo_id = "jakubvonasek/text2emoji"
    api = HfApi()
    
    try:
        create_repo(repo_id, exist_ok=True)
        print(f"created repo: {repo_id}")
    except Exception as e:
        print(f"repo exists or error: {e}")
    
    # upload files
    files_to_upload = [
        "model.py",
        "config.json", 
        "tokenizer_config.json",
        "semantic_emoji_model.pkl",
        "README.md",
        "requirements.txt",
        ".gitattributes"
    ]
    
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"uploaded: {file_path}")
        else:
            print(f"file not found: {file_path}")
    
    print("upload complete!")

if __name__ == "__main__":
    upload_model() 