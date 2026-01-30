# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Language model metadata curator that collects, validates, and curates information about transformer-based language models. Provides GUI tools for adding models and links, parsing arXiv papers, and generating markdown documentation.

## Common Commands

```bash
# Format and lint (primary command)
make all              # Runs format, lint_fix, lint_shell

# Individual commands
make lint             # Check with ruff
make format           # Format with ruff
make type_check       # mypy strict type checking
make lint_shell       # shellcheck for scripts

# Run GUI applications
uv run python -m list_lm.app                  # Main GUI for manual entry
uv run python -m list_lm.auto_add_lm_data_app # Semi-automated model addition (uses Ollama)
uv run python -m list_lm.auto_add_links_app   # Semi-automated link addition (uses Ollama)

# Merge JSON data
uv run python -m list_lm.merge_json lm <file.json>      # Merge models from JSON file
uv run python -m list_lm.merge_json links <file.json>   # Merge links from JSON file
uv run python -m list_lm.merge_json lm <file.json> --dry-run  # Preview without changes
```

Uses `uv` as package manager (not pip). Python 3.13+ required.

## Architecture

### Data Models (`list_lm/data.py`)
- `ModelInfo`: Language model metadata (name, year, publication URL, code, weights, video)
- `ApplicationData`: Curated resource links with `LinkType` categorization
- `ArticleData`/`ArticleDataExtended`: Paper metadata from arXiv parsing
- `LinkType`: 9 categories (MODEL, UTILS, GPU_PROFILING, VISUALIZATION, VOCABULARY, OPTIMIZER, DATASET, TOPIC, DOCUMENTATION)
- `UrlType`: URL classification (ARXIV, GITHUB, HUGGINGFACE, X/TWITTER, ACM, UNKNOWN)

### Core Modules
- `data_manager.py`: Generic CRUD with JSON persistence
- `merge_json.py`: Merge JSON data into LM data or links with duplicate detection
- `parse_html.py`: arXiv HTML parsing with cache (`.cache_arxiv.json`)
- `parse_url.py`: URL domain classification
- `parser_lm_data.py` / `parser_links.py`: Ollama LLM + regex extraction
- `generate_readme.py`: JSON to markdown conversion
- `validate_*.py`: Data validation modules
- `ollama_client.py`: Local Ollama LLM interface (http://localhost:11434)

### Data Storage
```
data/
├── json/
│   ├── model_data_list.json   # ModelInfo list
│   └── all_links.json         # ApplicationData list
├── readme/                    # Generated markdown files by category
└── .cache_arxiv.json          # arXiv parse cache
```

## Code Style

- Line length: 120 characters
- Strict mypy type checking enabled
- Ruff linting with comprehensive rules (E, F, W, I, N, ANN, S, B, etc.)
- All functions require type annotations

## Adding LM Data (ModelInfo)

When adding new language models, **always save to a separate JSON file first**, then merge using the merge script. This ensures proper validation and prevents accidental data loss.

### Workflow

1. Create a new JSON file (e.g., `new_models.json`) with a list of models
2. Preview the merge: `uv run python -m list_lm.merge_json lm new_models.json --dry-run`
3. Merge into main data: `uv run python -m list_lm.merge_json lm new_models.json`
4. The script automatically sorts and regenerates README files

### ModelInfo JSON Schema

```json
{
    "name": "ModelName",
    "year": 2024,
    "publication": {
        "title": "Paper Title",
        "url": "https://arxiv.org/abs/XXXX.XXXXX",
        "date_create": "YYYY-MM-DD"
    },
    "video": null,
    "code": null,
    "weights": null,
    "manual_validated": false
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model name (e.g., "GPT-4", "LLaMA 3") |
| `year` | int | Publication year |
| `publication` | ArticleData | Publication details (see below) |

### Choosing a Model Name

- If the paper introduces a named model, use that name (e.g., "Ministral 3", "LLaMA")
- If the paper describes a technique without a named model, ask the user what name to use
- Common naming patterns for technique papers:
  - Abbreviation of the method (e.g., "DropPE" for "Dropping Positional Embeddings")
  - Acronym from the paper title
  - Short descriptive name based on the core contribution

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `video` | UrlData or null | Video explanation link |
| `code` | UrlData or null | Source code repository |
| `weights` | UrlData or null | Model weights download |
| `manual_validated` | bool | Set to `true` after manual review |

### ArticleData Format (for publication)

```json
{
    "title": "Paper Title Here",
    "url": "https://arxiv.org/abs/XXXX.XXXXX",
    "date_create": "2024-01-15"
}
```

- `date_create` must be `YYYY-MM-DD` format
- For arXiv papers, use the original submission date

### Publication URL Requirements

**Always use specific publication URLs**, not general company or product pages:

- **Preferred**: arXiv paper, blog post announcing the model, or GitHub README
- **Avoid**: General company pages, product landing pages, or model collection pages

When given a general webpage that lists multiple models:
1. Check if each model has its own dedicated publication (arXiv paper, blog post, announcement)
2. If a model has its own publication, use that specific URL
3. Only use a general page if no specific publication exists for that model

Example of what to avoid:
- Using `https://company.com/models` (general models page) instead of `https://company.com/blog/announcing-model-x` (specific announcement)

### Custom Web Page Links (Blog Posts, Announcements)

When a model is announced via a blog post or company page instead of an arXiv paper:

1. Use `"Blog - ModelName"` as the title prefix
2. Use the blog/announcement URL as the publication URL
3. Extract code and weights links from the page if available

Example for a blog-based publication:

```json
{
    "name": "GLM-4.7",
    "year": 2025,
    "publication": {
        "title": "Blog - GLM-4.7",
        "url": "https://z.ai/blog/glm-4.7",
        "date_create": "2025-12-22"
    },
    "video": null,
    "code": {
        "title": "GitHub",
        "url": "https://github.com/zai-org/GLM-4.7"
    },
    "weights": {
        "title": "HuggingFace models",
        "url": "https://huggingface.co/zai-org/GLM-4.7"
    },
    "manual_validated": false
}
```

When fetching custom web pages, extract:
- Model name
- Release/announcement date (for `date_create`)
- GitHub/code repository URL
- HuggingFace/weights URL

### GitHub README as Publication

When a model is published via a GitHub repository without an arXiv paper or blog post:

1. Use `"README - ModelName repository"` as the title
2. Use the full GitHub URL pointing to a specific commit hash of the README file
3. The same repository URL (without commit hash) goes in the `code` field

Example for a GitHub-based publication:

```json
{
    "name": "OpenLLaMA",
    "year": 2023,
    "publication": {
        "title": "README - OpenLLaMA repository",
        "url": "https://github.com/openlm-research/open_llama/blob/6e7f73eab7e799e2464f38ed977e537bae02873e/README.md",
        "date_create": "2023-04-29"
    },
    "video": null,
    "code": {
        "title": "GitHub",
        "url": "https://github.com/openlm-research/open_llama"
    },
    "weights": {
        "title": "HuggingFace models",
        "url": "https://huggingface.co/openlm-research/open_llama_13b"
    },
    "manual_validated": false
}
```

To get the commit-specific URL:
1. Navigate to the README file in the repository
2. Click the "History" button or press `y` to get the permalink with commit hash
3. Use this URL as the publication URL

### Model Names with Acronyms/Shortcuts

When a paper introduces a model with both a full name and an acronym:

1. **If the acronym is more commonly known**: Use `"Acronym (Full Name)"` format
   - Example: `"KAN (Kolmogorov-Arnold Networks)"`
   - Example: `"CoPE (Contextual Position Encoding)"`

2. **If the full name is more commonly known**: Use `"Full Name (Acronym)"` format
   - Example: `"Byte Latent Transformer (BLT)"`
   - Example: `"BiT (Big Transfer)"`

The acronym/shortcut typically comes from the paper title or is explicitly defined in the paper. Include it to help users find the model by either name.

### Multiple Model Names

When a single paper introduces multiple distinct models or techniques:

Use `"Name1 / Name2"` format with spaces around the slash:
- Example: `"LightConv / DynamicConv"` for a paper introducing both techniques
- Example: `"BLOOMZ / mT0"` for related models from the same paper
- Example: `"24hBERT / Academic Budget BERT"` for a model known by multiple names

### UrlData Format (for video, code, weights)

```json
{
    "title": "GitHub",
    "url": "https://github.com/org/repo"
}
```

Common title conventions:
- Code: `"GitHub"`, `"GitLab"`, `"Bitbucket"`
- Weights: `"HuggingFace models"`, `"Model weights"`
- Video: `"YouTube"`, `"Video"`

### Sorting

Data is automatically sorted by `(publication.date_create, name.lower())`. Add entries in approximate order, validation will fix sorting.

### Searching for Code and Weights

After adding model entries, search the web for:
- **Code repository**: Search for GitHub/GitLab repositories (e.g., "ModelName GitHub repository")
- **Model weights**: Search for HuggingFace model weights (e.g., "ModelName HuggingFace weights")

Add found links to the `code` and `weights` fields. If not found, leave as `null`.

### After Adding

After merging, optionally run validation to check for issues:

```bash
uv run python -m list_lm.validate_lm_data
```

Do NOT run validation automatically. Wait for the user to tell you when to run it.

## Adding Links (ApplicationData)

When adding new resource links, **always save to a separate JSON file first**, then merge using the merge script.

### Workflow

1. Create a new JSON file (e.g., `new_links.json`) with a list of links
2. Preview the merge: `uv run python -m list_lm.merge_json links new_links.json --dry-run`
3. Merge into main data: `uv run python -m list_lm.merge_json links new_links.json`
4. The script automatically sorts and regenerates README files

### ApplicationData JSON Schema

```json
{
    "name": "ToolName",
    "description": "Brief description of what the tool does.",
    "url": "https://github.com/org/repo",
    "link_type": "Utils links",
    "manual_validated": false
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Tool/resource name |
| `description` | string | One-sentence description |
| `url` | string | Primary URL |
| `link_type` | LinkType | Category (see below) |

### LinkType Values

| Value | Description |
|-------|-------------|
| `"Model links"` | Model implementations, architectures |
| `"Utils links"` | Training utilities, inference tools, frameworks |
| `"GPU profiling links"` | GPU profiling and optimization tools |
| `"Visualization links"` | Visualization and interpretability tools |
| `"Vocabulary links"` | Tokenizers, vocabularies |
| `"Optimizer links"` | Optimizers, learning rate schedulers |
| `"Dataset links"` | Datasets, benchmarks, evaluations |
| `"Topic links"` | Research topics, papers, articles |
| `"Documentation links"` | Tutorials, documentation, guides |

### Sorting

Data is automatically sorted alphabetically by `name.lower()`.

### After Adding

After merging, optionally run validation to check for issues:

```bash
uv run python -m list_lm.validate_links
```

This checks:
- No duplicate names with same URL
- Warns about potential duplicates (same name, different URL)
- Correct sort order

Do NOT run validation automatically. Wait for the user to tell you when to run it.

## Regenerating README Files

After modifying JSON data, regenerate all markdown files:

```bash
uv run python -m list_lm.generate_readme
```

This creates/updates:
- `data/readme/language_models.md` (from model_data_list.json)
- `data/readme/*.md` files for each LinkType category (from all_links.json)
