# LMMs Engine Documentation

This directory contains the Sphinx-based documentation for LMMs Engine, configured for automatic building on ReadTheDocs.

## Documentation Structure

The documentation is organized into the following sections:

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── Makefile               # Build automation
├── requirements.txt       # Documentation dependencies
├── README.md              # This file
├── introduction.rst       # Introduction page
├── quick_start.rst        # Quick start guide
├── getting_started/       # Getting started section
│   └── index.rst
├── user_guide/            # User guides section
│   └── index.rst
├── developer_guide/       # Developer guide section
│   └── index.rst
├── reference/             # API reference section
│   └── index.rst
├── models/                # Model documentation
│   ├── index.rst
│   └── bagel.md
└── [markdown files]       # Content files (.md files)
    ├── train.md
    ├── datasets.md
    ├── data_prep.md
    ├── video_configuration.md
    ├── peak_perf.md
    ├── design_principle.md
    ├── new_model_guide.md
    ├── new_trainer_guide.md
    └── api.md
```

## Building Locally

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Other Build Formats

```bash
make clean      # Clean build artifacts
make dirhtml    # Build HTML in directory format
make latex      # Build LaTeX
make latexpdf   # Build PDF
make epub       # Build ePub
make linkcheck  # Check all external links
```

## ReadTheDocs Integration

This project is configured to automatically build on ReadTheDocs. The configuration is defined in:

- `.readthedocs.yml` - ReadTheDocs build configuration at the repository root
- `docs/conf.py` - Sphinx configuration

### Automatic Builds

ReadTheDocs will automatically:
1. Build the documentation when you push to the repository
2. Generate HTML, PDF, and ePub formats
3. Host the documentation at your ReadTheDocs URL

### Configuration

Key settings in `.readthedocs.yml`:
- Python 3.10
- Sphinx configuration from `docs/conf.py`
- PDF and ePub output formats enabled
- Uses `docs/requirements.txt` for dependencies

## Adding New Documentation

### Adding a New Page

1. Create a `.rst` or `.md` file in the appropriate section directory
2. Update the corresponding `index.rst` file in that section to include the new page

Example structure:

```rst
.. toctree::
   :maxdepth: 2

   existing_page
   new_page
```

### Markdown Support

The documentation supports both reStructuredText (`.rst`) and Markdown (`.md`) formats thanks to the `myst-parser` extension.

For Markdown files, use standard Markdown syntax. For cross-references, use Sphinx directives like:
- `:doc:\`path/to/file\`` - Link to another documentation file
- `:ref:\`label\`` - Link to a label
- `{doc}\`path/to/file\`` - MyST syntax for document links

### Themes and Customization

The documentation uses the **Read the Docs** Sphinx theme (`sphinx-rtd-theme`) with customizations in `conf.py`.

To customize the theme, edit the `html_theme_options` in `conf.py`:

```python
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'style_nav_header_background': '#2980B9',
}
```

## Documentation Guidelines

### File Naming

- Use lowercase with underscores: `new_model_guide.md`
- For sections, use `index.rst`

### Section Organization

- **Getting Started**: Installation and basic setup
- **User Guide**: How to use various features
- **Developer Guide**: Architecture, extending, contributing
- **Reference**: API documentation
- **Models**: Specific model documentation

### Writing Style

- Use clear, concise language
- Include code examples where helpful
- Cross-reference related documentation
- Keep markdown content unchanged; only reorganize structure

## Troubleshooting

### Build Fails

1. Ensure all dependencies are installed: `pip install -r docs/requirements.txt`
2. Check for broken cross-references: `make linkcheck`
3. Clear build cache: `make clean && make html`

### Missing Files

If a referenced file is missing, update the `toctree` entries in the corresponding `index.rst` file.

### Preview Locally

After building, use a local HTTP server to preview:

```bash
cd docs/_build/html
python -m http.server 8000
```

Then visit `http://localhost:8000` in your browser.

## Next Steps

1. Connect this repository to ReadTheDocs
2. Set up webhooks for automatic builds
3. Configure custom domain (if applicable)
4. Enable PDF/ePub downloads

For more information, visit [Sphinx Documentation](https://www.sphinx-doc.org/) or [ReadTheDocs Documentation](https://docs.readthedocs.io/).
