# Recorded Harbor output

`real_job/` is a verbatim copy of a real `harbor run` job directory (harbor
0.20.0, the smoke tier, one task), trimmed to the two `result.json` files
`evals/scripts/gate.py` reads.

It is here because the gate's fabricated fixtures cannot pin a schema on their
own. The gate originally looked for a `trial_results` list on the job's
`result.json`; every hand-written test agreed with it, and the first real job
failed to grade after a run that had already scored 1.0. Rewards actually live in
`<job>/<trial>/result.json`.

Re-record it after bumping `HARBOR_VERSION`, by copying the two files out of a
fresh `jobs/<timestamp>/`. If the gate then fails on it, the schema moved and the
gate has to move with it — which is the whole point of keeping this.
