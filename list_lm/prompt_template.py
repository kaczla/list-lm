from string import Template

MODEL_NAME_PROMPT_TEMPLATE = Template(
    """You are AI assistance which responses for user questions.

Here is a title of publication:
$title

And here is the abstract of the publication:
$abstract

Tell me what is the name of the model or presented solution in this publication?
Respond only name without additional text. Do not normalize the name, return original name.
"""
)


def get_model_name_prompt(title: str, abstract: str) -> str:
    prompt = MODEL_NAME_PROMPT_TEMPLATE.safe_substitute({"title": title, "abstract": abstract})
    return prompt
