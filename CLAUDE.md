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

# Validation (run only when user requests)
uv run python -m list_lm.validate_lm_data     # Validate model data
uv run python -m list_lm.validate_links       # Validate links data
```

**Important:** After creating a new JSON file with model or link data, do NOT automatically run or propose running `merge_json`. Wait for the user to request it.

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

## Adding Data (General Workflow)

Both model data and links follow the same workflow:

1. Create a new JSON file with timestamp (e.g., `new_models_20250130_1430.json`)
2. Merge into main data using the merge script
3. The script automatically sorts and regenerates README files

**Important:** Do NOT run validation automatically. Wait for the user to request it.

## Adding LM Data (ModelInfo)

### Merge Command

```bash
uv run python -m list_lm.merge_json lm <file.json>
```

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

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `video` | UrlData or null | Video explanation link |
| `code` | UrlData or null | Source code repository |
| `weights` | UrlData or null | Model weights download |
| `manual_validated` | bool | Set to `true` after manual review |

### Model Naming Rules

**Basic naming:**
- If the paper introduces a named model, use that name (e.g., "Ministral 3", "LLaMA")
- If the paper describes a technique without a named model, ask the user what name to use

**With acronyms:** Use `"Full Name (Acronym)"` format
- `"Kolmogorov-Arnold Networks (KAN)"`
- `"Contextual Position Encoding (CoPE)"`

**Multiple models in one paper:** Use `"Name1 / Name2"` format
- `"LightConv / DynamicConv"` for a paper introducing both techniques
- `"BLOOMZ / mT0"` for related models from the same paper

**Technique papers without a model name:**
- Abbreviation of the method (e.g., "DropPE" for "Dropping Positional Embeddings")
- Acronym from the paper title
- Short descriptive name based on the core contribution

### Publication Sources (Priority Order)

Use the first available source:

| Priority | Source | Title Format | Example |
|----------|--------|--------------|---------|
| 1 | arXiv paper | Paper title | `"Attention Is All You Need"` |
| 2 | Blog post | `"Blog - ModelName"` | `"Blog - GLM-4.7"` |
| 3 | GitHub README | `"README - ModelName repository"` | `"README - OpenLLaMA repository"` |
| 4 | HuggingFace | `"HuggingFace - ModelName"` | `"HuggingFace - Dolphin3.0-Llama3.2-1B"` |

**Publication URL must be specific.** Never use general company or product pages.

| Correct | Incorrect |
|---------|-----------|
| `https://company.com/blog/announcing-model-x` | `https://company.com/models` |
| `https://arxiv.org/abs/2301.00001` | `https://company.com/research` |

### ArticleData Format

```json
{
    "title": "Paper Title Here",
    "url": "https://arxiv.org/abs/XXXX.XXXXX",
    "date_create": "2024-01-15"
}
```

- `date_create` must be `YYYY-MM-DD` format
- For arXiv papers, use the original submission date

### Publication Examples by Source Type

**Blog post:**
```json
{
    "name": "GLM-4.7",
    "year": 2025,
    "publication": {
        "title": "Blog - GLM-4.7",
        "url": "https://z.ai/blog/glm-4.7",
        "date_create": "2025-12-22"
    },
    "code": {"title": "GitHub", "url": "https://github.com/zai-org/GLM-4.7"},
    "weights": {"title": "HuggingFace models", "url": "https://huggingface.co/zai-org/GLM-4.7"}
}
```

**GitHub README:** Use commit-specific URL (press `y` on GitHub to get permalink)
```json
{
    "name": "OpenLLaMA",
    "year": 2023,
    "publication": {
        "title": "README - OpenLLaMA repository",
        "url": "https://github.com/openlm-research/open_llama/blob/6e7f73eab7e799e2464f38ed977e537bae02873e/README.md",
        "date_create": "2023-04-29"
    },
    "code": {"title": "GitHub", "url": "https://github.com/openlm-research/open_llama"},
    "weights": {"title": "HuggingFace models", "url": "https://huggingface.co/openlm-research/open_llama_13b"}
}
```

**HuggingFace (last resort):**
```json
{
    "name": "Dolphin 3.0 Llama 3.2 1B",
    "year": 2025,
    "publication": {
        "title": "HuggingFace - Dolphin3.0-Llama3.2-1B",
        "url": "https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.2-1B",
        "date_create": "2025-01-05"
    },
    "weights": {"title": "HuggingFace models", "url": "https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.2-1B"}
}
```

### Single Entry vs Multiple Entries

**Decision tree for multiple models:**

```
Webpage lists multiple models
    │
    ├─► Do they have individual publications (arXiv, blog posts)?
    │       YES → Create separate entries with specific URLs
    │       NO  ↓
    │
    ├─► Are they size variants only (1B, 3B, 8B)?
    │       YES → Single entry (e.g., "Model") with collection URL
    │       NO  ↓
    │
    └─► Are they different types (text vs vision)?
            YES → Separate entries OR combined name "Model-Text / Model-Vision"
```

**Single entry:** Same architecture, different sizes only
```json
{"name": "Granite 3.1", "weights": {"url": "https://huggingface.co/collections/ibm-granite/..."}}
```

**Multiple entries:** Different capabilities or separate releases
- `"LLaMA 3"` and `"LLaMA 3.1"` (different releases)
- `"MiniMax-Text-01"` and `"MiniMax-VL-01"` (different capabilities)

When uncertain, **ask the user**.

### UrlData Format (for video, code, weights)

```json
{"title": "GitHub", "url": "https://github.com/org/repo"}
```

| Field | Common titles |
|-------|---------------|
| code | `"GitHub"`, `"GitLab"`, `"Bitbucket"` |
| weights | `"HuggingFace models"`, `"Model weights"` |
| video | `"YouTube"`, `"Video"` |

### Weights URL Requirements

| Use case | URL type | Example |
|----------|----------|---------|
| Model family (1B/3B/7B variants) | Collection URL | `https://huggingface.co/collections/deepseek-ai/deepseek-coder` |
| Specific version (Eagle 2 vs 2.5) | Specific model URL | `https://huggingface.co/nvidia/Eagle2-1B` |
| Single model, no family | Specific model URL | `https://huggingface.co/google/vaultgemma-1b` |
| Only org page available | `null` | Never use `https://huggingface.co/nvidia` |

### Searching for Code and Weights

After adding model entries, search the web for:
- **Code repository**: "ModelName GitHub repository"
- **Model weights**: "ModelName HuggingFace weights"

Add found links to `code` and `weights` fields. If not found, leave as `null`.

### Sorting

Data is automatically sorted by `(publication.date_create, name.lower())`.

## Adding Links (ApplicationData)

### Merge Command

```bash
uv run python -m list_lm.merge_json links <file.json>
```

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

### Naming Conventions

When a resource has both a full name and an acronym, use `"Full Name (Acronym)"` format:
- `"Preconditioned Stochastic Gradient Descent (PSGD)"`

### LinkType Values

| Value | Description |
|-------|-------------|
| `"Model links"` | Model implementations, architectures |
| `"Utils links"` | Training utilities, inference tools, frameworks |
| `"GPU profiling links"` | GPU profiling and optimization tools |
| `"Visualization links"` | Visualization and interpretability tools |
| `"Vocabulary links"` | Tokenizers, vocabularies |
| `"Optimizer links"` | Optimizers, LR schedulers, optimization techniques |
| `"Dataset links"` | Datasets, benchmarks, evaluations |
| `"Topic links"` | Research topics, papers, articles |
| `"Documentation links"` | Tutorials, documentation, guides |

### Sorting

Data is automatically sorted alphabetically by `name.lower()`.

## Regenerating README Files

After modifying JSON data, regenerate all markdown files:

```bash
uv run python -m list_lm.generate_readme
```

This creates/updates:
- `data/readme/language_models.md` (from model_data_list.json)
- `data/readme/*.md` files for each LinkType category (from all_links.json)
