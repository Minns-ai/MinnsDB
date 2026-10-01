# Support time machine

Support windows for 36 runtimes, databases and operating systems, stored as temporal graph edges. Pick a date and one MinnsQL query returns every release that was supported that day.

It shows the part of agent memory that flat stores get wrong: facts that stop being true. "Node.js 16 is supported" was true from 20 Apr 2021 to 11 Sept 2023. Here it is an edge with `valid_from` and `valid_until`, so "was it supported on this date" is a `WHEN` filter.

No LLM is involved. The data goes in through `POST /api/graph/import`.

## Files

| File | What it does |
|---|---|
| `load.py` | Builds nodes and edges from `eol/` and imports them in one request. Standard library only. |
| `index.html` | A page with a date slider. Each change runs one MinnsQL query against the server. |
| `eol/` | Snapshot of the [endoflife.date](https://endoflife.date) API, fetched 2026-10-01 (MIT, see `eol/LICENSE`). |
| `fetch_data.py` | Refreshes the snapshot. |

## Run it

Start MinnsDB (see the main [README](../../README.md#quick-start)), then from this directory:

```bash
python3 load.py
# 784 nodes, 1123 edges
# {"nodes_created":784,"nodes_reused":0,"edges_created":1123,"errors":[]}

python3 -m http.server 8093
# open http://localhost:8093
```

`load.py` reads `MINNS_URL` (default `http://localhost:3000`) and, if authentication is on, `MINNS_KEY`. The page takes `?api=http://host:port` to point at another server and `?date=YYYY-MM-DD` to open on a date. It calls the API from the browser, so it needs CORS, which is permissive by default when authentication is off.

With Docker Compose, set `QDRANT_API_KEY` to any value. If it is unset, Compose passes an empty key to Qdrant and the server exits on boot with "The request does not have valid authentication credentials".

## The model

```
(Node.js)-[supports]->(Node.js 16)        releaseDate .. eol
(Node.js)-[active_support]->(Node.js 16)  releaseDate .. end of active support
(Node.js)-[lts]->(Node.js 16)             lts date .. eol
```

A release with `eol: false` (still supported) becomes an open-ended edge, with `valid_until` null. Releases with no exact dates are skipped.

The page runs:

```
MATCH (p)-[r]->(v) WHEN "2023-09-10"
RETURN v.product, v.cycle, r.association_type, valid_until(r)
```

## Results to check against

- `WHEN "2019-12-31"` returns Python 3.8, 3.7, 3.6, 3.5 and 2.7. `WHEN "2020-01-01"` returns the same without 2.7.
- `WHEN "2023-09-10"` returns Node.js 20, 18 (LTS, active support) and 16 (LTS). `WHEN "2023-09-11"` returns 20 and 18.
- `valid_from` is inclusive and `valid_until` is exclusive. Python 3.0 is in the result on its release day, 2008-12-03.
- In a local Docker run the server reported `execution_time_ms` of 0 to 1 once warm, and 3 to 6 on the first query after a restart.

## Limits

- `valid_from` and `valid_until` are unsigned nanoseconds since 1970, so dates before 1970 can't be stored.
- endoflife.date doesn't cover every product from its first release (Angular starts at v9, 2020). The page shows "no data before" for those periods.
- Future end-of-life dates in the snapshot can change.
