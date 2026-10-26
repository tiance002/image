# Bus/train supervision and miss diagnosis (NP2-R)

This is the curated public evidence package for the frozen-checkpoint bus/train diagnostic, based on `np2r-revision-20261026` at `fa209be9036a9619d9298535470b51574bccd73a`.

Start with `BUS_TRAIN_DIAGNOSIS_REPORT_v2.md`. The report marks the overall root cause `BLOCKED_ROOT_CAUSE_UNRESOLVED`; its proposed native TaskAlignedAssigner stage trace was not run. See `REPORT_STATUS.md` for superseded local drafts and `REPRODUCE_v2.md` for the audit protocol.

The package contains derived tables, diagnostic JSON, and source for the recorded offline/gradient checks. It intentionally excludes original images, model checkpoints, formal prediction tensors, local environment/path inventories, and intermediate failed-run logs. The dataset, checkpoints, and original predictions are not distributed here; these outputs are not independently reproducible without the authorized local research assets.
