# DeepWiki August Release Notes

**Date:** 2026-08-20
**Branch:** `fix/xixia/aml-forcerun`
**Period covered:** 2026-06-01 → 2026-08-20 (since [release-202605.md](./release-202605.md))

---

## Highlights

- **Deterministic first AML run** — new schedules explicitly submit one pipeline job instead of relying on a provisioning-time recurrence trigger.
- **`--run-now` dispatcher flag** — existing scheduled pipelines can be submitted immediately while retaining their recurring schedule.
- **Stable recurrence timing** — the next automatic run starts one configured interval after schedule reconciliation, preventing a duplicate run near the explicit submission.

---

## Azure ML Pipeline Scheduling

The AML dispatcher now separates immediate job submission from recurring schedule execution:

- A new schedule submits exactly one immediate pipeline job.
- An existing schedule submits immediately only when `--run-now` is specified.
- The recurring trigger starts at UTC now plus `azure_ml.schedule_interval_hours`.
- Schedule lookup treats only `ResourceNotFoundError` as a missing schedule; authentication and service errors propagate.
- Existing schedules are updated in place, so a rejected update no longer disables the previous schedule.
- A disabled schedule is re-enabled only after the replacement schedule is accepted.
- Immediate submission failures preserve the configured schedule and report that the operator can retry with `--run-now`.
- The cloud resource summary includes `aml_job` when an immediate job is submitted.

### Run an existing schedule now

```bash
python -m backend.processor.aml_dispatcher \
    --config=backend/run.json --run-now
```

The flag controls AML job submission only. It is separate from `code_processor --full-reprocess`, which controls incremental content processing inside a running job.

---

## Documentation

- Updated cloud-processing examples in the root, backend, and processor READMEs.
- Clarified the difference between immediate AML submission and full content reprocessing.

---

## Files Changed (Highlights)

| File | Change |
|------|--------|
| `backend/processor/aml_dispatcher.py` | Added `--run-now` and passed the submission intent into cloud setup |
| `backend/processor/cloud_setup.py` | Added deterministic recurrence timing, explicit immediate submission, narrow schedule lookup handling, and submitted-job reporting |
| `README.md` | Documented new-schedule and existing-schedule commands |
| `backend/README.md` | Documented immediate cloud pipeline submission |
| `backend/processor/README.md` | Distinguished `--run-now` from `--full-reprocess` |
| `Note/release-202608.md` | This document |
