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

LINKS_PROMPT_TEMPLATE = Template(
    """You are AI assistance which responses for user questions.

Here is a markdown README:
$readme_text

Here is the end of the README.

Tell me:
1. What is the name of the application in this repository.
2. Provide a short description of the application.
3. Choose one category that describes this application from the available list (separated by commas): $categories_text
Respond as the JSON data only without any additional text or explanation.

Here is the example response:
{
  "name": "nanoGPT",
  "description": "nanoGPT is the simplest, fastest repository for training/fine-tuning medium-sized GPTs.",
  "category": "MODEL"
}
where:
- "name" correspond to application name,
- "description" to description of application,
- "category" is the selected category.

Remember do not normalize the name, return original name as it is.
"""
)


def get_model_name_prompt(title: str, abstract: str) -> str:
    prompt = MODEL_NAME_PROMPT_TEMPLATE.safe_substitute({"title": title, "abstract": abstract})
    return prompt


def get_links_prompt(readme_text: str, categories_text: str) -> str:
    prompt = LINKS_PROMPT_TEMPLATE.safe_substitute({"readme_text": readme_text, "categories_text": categories_text})
    return prompt
