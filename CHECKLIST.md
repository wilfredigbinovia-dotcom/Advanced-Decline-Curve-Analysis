# Deployment checklist

Your repository root must contain **all four** of these files, side by side:

```
advanced-decline-curve-analysis/
├── app.py                  <- Streamlit entry point
├── dca_charts.py           <- Plotly charts        (imported by app.py)
├── gas_condensate_dca.py   <- analysis engine      (imported by both)
├── requirements.txt        <- MUST be at the root
└── .streamlit/
    └── config.toml         <- optional, theme only
```

Verify from a terminal in the repo:

```bash
ls app.py dca_charts.py gas_condensate_dca.py requirements.txt
git status --short          # anything untracked is not deployed
```

An untracked or uncommitted file does not exist as far as Streamlit Cloud is
concerned, which is the usual cause of a `ModuleNotFoundError` that names one
of your own files.

## Reading the two failure modes

| Traceback frame | Meaning | Fix |
|---|---|---|
| inside `gas_condensate_dca.py`, on `from scipy import ...` | a third-party package is not installed | add it to `requirements.txt` at the repo root |
| inside `app.py`, on `import gas_condensate_dca` | one of your own files is missing from the repo | commit `gas_condensate_dca.py` next to `app.py` |

Streamlit Cloud redacts the error text, so the **file and line in the traceback
is the only signal you get**. The frame tells you which of the two it is.

After any change: **Manage app -> Reboot app**. A cached environment sometimes
survives a plain commit.

## Local smoke test before pushing

```bash
pip install -r requirements.txt
python gas_condensate_dca.py --selftest   # expect 20/20 passed
streamlit run app.py
```
