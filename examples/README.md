# examples/corpus

Three documents, one of them in a subdirectory, written so the quickstart in
the top-level README produces the same counts every time.

`payouts/schedule.md` is there on purpose: the doc_id is the path relative to
the corpus root, `payouts/schedule.md`, so a file moved between directories is
a delete plus an add rather than a change. That is the right answer for a
corpus where the path carries meaning, and the wrong one for a corpus where it
does not. If paths in your corpus are incidental, hash the content into the
doc_id instead and the move becomes free.
