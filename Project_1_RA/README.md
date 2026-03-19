# Project 1 RA

A simple research assistant CLI using Gemini and Google Search.

## Prerequisites

- Python 3.11+
- A Gemini API key

## Setup (Windows PowerShell)

```powershell
cd "c:\Yaswanth files\AI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\Project_1_RA\requirements.txt
```

## Configure API Key

Set your Gemini API key in the current terminal session:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

## Run

```powershell
python .\Project_1_RA\agent.py
```

Type `quit` to exit.

## Notes

- Use `python -m pip` to ensure packages install into the active environment.
- The `.venv` folder is ignored by Git and should not be committed.
