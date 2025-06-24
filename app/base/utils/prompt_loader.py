import os

def load_prompt_template(path: str) -> str:
    """
    Load raw prompt from .txt file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(template: str, context: dict) -> str:
    """
    Render the prompt using Python's str.format with the given context.
    """
    try:
        return template.format(**{k: str(v) for k, v in context.items()})
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")
