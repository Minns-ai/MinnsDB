"""Refresh the endoflife.date snapshot in eol/. The committed copy was fetched on 2026-10-01."""
import os, urllib.request
from load import NAMES

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eol")
for slug in NAMES:
    url = f"https://endoflife.date/api/{slug}.json"
    with urllib.request.urlopen(url, timeout=30) as r, open(os.path.join(out, slug + ".json"), "wb") as f:
        f.write(r.read())
    print("fetched", slug)
