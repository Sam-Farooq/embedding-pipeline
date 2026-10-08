# embedding-pipeline

A batch job that keeps a Qdrant collection in step with a directory of
documents. It hashes the corpus, works out what actually changed, embeds only
that, and deletes the vectors whose source document is gone.

State lives in one JSON file. Nothing else is remembered between runs.

## The offline quickstart

No model download, no Qdrant, no network. `--embedder hash` produces
deterministic vectors with no meaning in them, and `--store jsonl` writes
points to a local file, so the mechanism can be watched end to end.

The three documents under `examples/corpus` are invented sample text, written
for this repo so that the counts below come out the same on any machine.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp -r examples/corpus /tmp/corpus

embedpipe run --corpus /tmp/corpus --embedder hash --store jsonl
```

```
documents embedded            3
chunks embedded               4
batches                       1
chunks over the token budget  0
documents payload-only        0
payload writes                0
documents removed             0
points deleted                0
documents unchanged           0
manifest saves                2
```

Now change one document and run it again:

```bash
echo "A refund after 180 days is refused by the network, not by us." >> /tmp/corpus/refunds.md
embedpipe run --corpus /tmp/corpus --embedder hash --store jsonl
```

```
documents embedded            1
chunks embedded               2
batches                       1
...
documents unchanged           2
```

Two of three documents were skipped on their content hash. The third was
re-chunked and re-embedded whole, which is the first tradeoff below.

## What the manifest records, and why

`.embedpipe/manifest.json` after the first of those two runs, one record per
document:

```json
"refunds.md": {
  "chunk_count": 2,
  "content_hash": "73fd94eb78444358",
  "embedded_at": "2026-10-08T12:17:34+00:00",
  "model": "hash-not-a-model@v1/d384",
  "path": "/tmp/corpus/refunds.md",
  "payload_signature": "d45ebfc61be2a061"
}
```

Three of those fields each answer a question that content hashing alone cannot:

- **`model`.** A model swap invalidates every vector in the collection, whatever
  the documents say. Cosine distance between a bge vector and an e5 vector is a
  number, which is worse than an error, because search keeps working and the
  results quietly stop meaning anything. The fingerprint is recorded per
  document rather than once per collection, so a run interrupted halfway through
  a model migration leaves a manifest that says exactly which documents made it
  across.
- **`payload_signature`.** A hash of the payload field names plus the static
  values passed with `--set`. It covers everything except `ingested_at`, because
  a timestamp inside the signature would mark the whole corpus dirty on every
  run.
- **`chunk_count`.** Point ids are `uuid5(namespace, "<doc_id>#<index>")`, so a
  document's ids are derivable from its id and its count. The manifest stores no
  id lists, and it cannot disagree with itself about which points a document
  owns.

## Six outcomes per document

| Outcome | Condition | Cost |
|---|---|---|
| unchanged | same content, model and payload signature | nothing, not even chunking |
| new | not in the manifest | embed every chunk |
| changed | content hash differs | embed every chunk, delete any orphans |
| model-changed | content same, model fingerprint differs | embed every chunk |
| payload-only | content and model same, payload signature differs | **no forward pass** |
| removed | in the manifest, not in the corpus | delete its points |

`embedpipe plan` prints the breakdown and writes nothing. With
`--fail-on-drift` it exits 3 when work is pending, which is the form a cron job
or a CI check wants.

## Adding a field to a million vectors without re-embedding them

Chunking is string work and the forward pass is not, so a payload change
reproduces the chunk text from the corpus and rewrites payloads over the
vectors that are already there:

```bash
embedpipe backfill --corpus /tmp/corpus --embedder hash --store jsonl --set source=handbook
```

```
documents embedded            0
chunks embedded               0
documents payload-only        3
payload writes                4
```

Zero chunks embedded, four payloads written. `backfill` refuses outright if any
document needs embedding, so adding a field cannot turn into an unplanned
rebuild because somebody edited a file that morning.

## Surviving a kill

The rule is that **a document is written to the manifest only once every one of
its chunks has been accepted by the store.** Batches span documents, because
batching per document wastes the budget on a corpus of short files, so a batch
dying halfway leaves one document part-written. That document is absent from the
manifest and the next run embeds it again from chunk zero.

Re-embedding cannot duplicate anything, because the point ids are derived
rather than random, so the second upsert overwrites the first. There is no
cleanup pass and no dangling-vector sweep.

`tests/test_pipeline.py::test_a_crash_mid_document_does_not_half_commit` is the
test that holds this: a store that raises on its third upsert, one document of
one chunk and one of three, then a resume that embeds three chunks rather than
four and leaves four points rather than five.

Two layers, and they cover different things. A `try/finally` saves the manifest
when an exception escapes. `--commit-every N` bounds what is lost when the
process is killed outright and no handler runs at all. The default of 1 flushes
after every batch, which costs an fsync per batch and loses at most one batch of
work. `--commit-every 10` is the other end of that.

## Batching against memory, not against a number

A fixed batch size is sized for the worst chunk in the corpus or it is wrong.
Thirty-two chunks of 40 tokens and thirty-two of 512 are the same item count
and nearly thirteen times the tokens, and it is the token count that decides
whether the job survives, because that is what activation memory follows. A
batch therefore closes when either cap is reached: 32 items or 8,192 tokens.

Token counts come from `CharCounter` by default, which is length over four. That
rule of thumb over-counts code and under-counts agglutinative languages, and
being wrong costs a batch of the wrong size rather than a wrong answer.
`--token-counter tokenizer` loads the real Hugging Face tokenizer, a few
megabytes rather than the model's gigabytes, when the corpus is mixed enough to
need it.

A chunk whose own token count exceeds the budget is sent in a batch of one and
counted in `chunks over the token budget`. Dropping it would lose part of the
corpus with nothing to show that it happened.

## Against a real model and a real Qdrant

```bash
pip install -e ".[dev,hf]"
docker run -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.11.3

embedpipe run \
  --corpus ./docs \
  --embedder sentence-transformers \
  --model BAAI/bge-small-en-v1.5 \
  --dim 384 \
  --store qdrant --qdrant-url http://localhost:6333
```

`sentence-transformers` and `transformers` sit in the `hf` extra on purpose. The
base install is enough for `plan`, `status`, `report` and `verify`, and asking
for a 2 GB torch download to find out whether anything changed is the wrong
trade.

Add `--revision` with the commit sha from the model's page on the Hub. Without
it the manifest records the revision as `unpinned`, and an upstream change to
the weights then produces vectors from a different model that nothing in this
pipeline can detect. The
model identity is derived from the flags rather than read back off the loaded
model, so `plan` knows which model a run would use without downloading it; a
`--dim` that disagrees with the loaded model is an error at startup.

A width change is refused before anything is written:

```
embedpipe: manifest records 384-dimensional vectors from hash-not-a-model@v1/d384,
this run would write 768 from hash-not-a-model@v1/d768. Re-run with --recreate
to drop the collection and rebuild it.
```

Qdrant rejects a point of the wrong width, so without that check the run would
fail at the first upsert with every old vector still in place and the manifest
claiming they were current.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | nothing went wrong |
| 2 | refused: bad flags, an unreadable manifest, a width change without `--recreate` |
| 3 | `plan --fail-on-drift` found work to do |
| 4 | `verify` found the manifest and the store disagreeing |

`verify` compares the ids the manifest implies against the ids the store holds
and reports both directions. It scrolls with `with_vectors=False`, so checking a
collection does not cost more than building it.

## Tests

```bash
pytest -q        # 90 tests, no network, no Qdrant, no model
```

The embedder and the store both sit behind protocols with working fakes, so
everything above is reachable offline. The Qdrant adapter is tested by injecting
a fake client and asserting the request shapes: derived ids, vector width,
payload contents, `wait=True` on every write, and that a collection is created
once rather than on every run.

`wait=True` is not decoration. A fire-and-forget upsert lets the manifest record
work the server has not accepted, and the manifest is the only record of what
has been embedded, so it has to be the pessimistic one.

## What this gave up

- **Whole-file hashing.** A one-word edit to a 40 page document re-embeds all of
  it. Per-chunk hashing would fix that, and it would also mean a paragraph
  inserted near the top shifts every later chunk, which under index-based ids
  invalidates them anyway. Stable ids and a manifest small enough to read in a
  terminal were worth more than fine-grained detection here. For a corpus of
  long, rarely-edited documents the trade goes the other way.
- **Index-based point ids.** Cheap, idempotent, and tied to position. Content
  addressed chunk ids would survive an insertion, at the cost of a second index
  to answer "which points does this document own".
- **One JSON file, read and rewritten whole.** The three-document example
  manifest is 1,007 bytes, and dropping one document took it to 733, so a record
  is about 275 bytes. By that arithmetic 100,000 documents is a 27 MB file
  parsed into memory and rewritten on every run. That is fine, and it stops
  being fine somewhere in the low millions, where this wants to be a table in
  Postgres instead.
- **One writer, no concurrency.** A thread pool over the batches would raise
  throughput and would make "a document commits only when all of its chunks
  land" much harder to be sure of.
- **One payload request per chunk.** The text and the index differ per point and
  Qdrant applies one payload per call, so a backfill of N chunks is N requests.
  Splitting the static fields from the per-chunk ones would cut that roughly in
  half and is not done.
- **No reranking, no retrieval, no search.** This produces and maintains
  vectors. Querying them is somebody else's repo.

## Known rough edges

- `backfill` assumes the points it is updating exist. If they do not, Qdrant
  accepts the call and nothing happens, and `verify` is how you find out.
- A file moved between directories is a delete plus an add, because the doc_id
  is the path relative to the corpus root. That is right where paths carry
  meaning and wrong where they do not.
- `--glob` defaults to `**/*.md`. There is no extractor for PDF or HTML, and
  anything that is not UTF-8 is decoded with replacement characters and hashed
  on its original bytes.
