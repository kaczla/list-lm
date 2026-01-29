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

To add a new language model entry, edit `data/json/model_data_list.json` directly.

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

Do NOT run validation automatically. Wait for the user to tell you when to run it.

```bash
uv run python -m list_lm.validate_lm_data
```

## Adding Links (ApplicationData)

To add a new resource link, edit `data/json/all_links.json` directly.

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

Run validation to check for errors and regenerate markdown:

```bash
uv run python -m list_lm.validate_links
```

This checks:
- No duplicate names with same URL
- Warns about potential duplicates (same name, different URL)
- Correct sort order

## Regenerating README Files

After modifying JSON data, regenerate all markdown files:

```bash
uv run python -m list_lm.generate_readme
```

This creates/updates:
- `data/readme/language_models.md` (from model_data_list.json)
- `data/readme/*.md` files for each LinkType category (from all_links.json)
