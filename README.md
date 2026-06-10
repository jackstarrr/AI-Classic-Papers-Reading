# AI Classic Papers Reading

A curated reading library of 30 influential AI papers, with Chinese reading notes and runnable toy code examples for understanding the core mechanisms.

This repository does not redistribute paper PDFs. Each paper directory includes `source.md` with the original source link.

## What Is Included

- `00_论文清单.csv`: the full paper list, categories, source links, and code themes.
- `ai_paper_examples.py`: shared implementations for all runnable examples.
- `NN_short_name/论文解读.md`: Chinese reading notes for each paper.
- `NN_short_name/source.md`: original paper source link.
- `NN_short_name/code/run.py`: a runnable minimal example.
- `NN_short_name/code/README.md`: how to run the example.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your current `python3` already has NumPy installed, using a virtual environment is optional.

## Run Examples

Run one paper example:

```bash
python3 22_transformer_2017/code/run.py
```

Run all shared examples:

```bash
python3 ai_paper_examples.py
```

## Directory Layout

```text
NN_short_name/
  source.md
  论文解读.md
  code/
    run.py
    README.md
```

## Notes

The examples are intentionally small. They are designed to expose the mechanism behind each paper, not to reproduce full-scale experiments.

Paper PDFs are excluded to avoid copyright and redistribution issues. Please use the links in `source.md` to access the original papers.
