import json

def load_config(filename="config.json"):
    """
    Load config file
    """
    f = open(filename, encoding="utf-8")
    return json.load(f)