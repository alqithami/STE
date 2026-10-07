# STE UC equal-compute study: reproduction package

The original prospective package was executed on an IBM NVIDIA L40S. The
production study completed on 5 October 2026; its outcomes are reported separately
from the original matched-epoch study in the revised manuscript. This source copy
contains the unchanged scientific runner, analysis, protocol, configuration,
requirements, and vendored operators. It contains no weights or training outputs.

Two instruction files, this README and LAUNCH_AND_SHARE.md, were updated to report
completion and remove personal server-login details. The previous paper PDF and
preparation-only software QA records were omitted. ANONYMIZATION_AND_PROVENANCE.json
records these changes; frozen/FROZEN_SHA256SUMS.txt is unchanged.

## Recorded study and outcome

- Seed 2026100407; 12 independent collections with 3 paired initializations.
- UC training/development at n=12; held-out tests at n=24 and n=48.
- Four systems: pairwise-only structural UC, ordinary auxiliary direct head,
  relational auxiliary direct head, and STE structural UC.
- Each system receives 220 seconds of measured training/development work per
  collection and initialization. The supervised systems split this between two
  110-second loss-weight candidates. Five development checkpoints and threshold
  selection are charged to the candidate budget. Every measured candidate and
  system total passed the frozen +/-2% tolerance.
- Primary endpoint: selective non-singleton UC F1 at n=24, two collection-level
  paired contrasts with Holm correction. Initializations are averaged within
  collections; cases and initializations are not independent inference units.
- Primary means: STE .6281, ordinary head .6414, relational head .6416.
  STE-minus-head differences are -.0133 and -.0134; both Holm p=.3193. This study
  demonstrates neither superiority, inferiority, equivalence nor noninferiority.
- Allocation was 31,680 seconds; charged work was 31,713.3455 seconds. The scientific
  invocation lasted 32,050.6229 seconds, excluding subsequent analysis/packaging.

PROTOCOL.md and config.json retain the pre-execution scientific specification.
Prospective wording in the immutable protocol and frozen/PROVENANCE.txt documents
what was known at preparation time; it does not describe the study's current status.

## Fresh reproduction

Follow [LAUNCH_AND_SHARE.md](LAUNCH_AND_SHARE.md). Linux, a compatible NVIDIA GPU
and Python 3.10--3.12 are required (3.11 preferred). The recorded run used Python
3.11.17, torch 2.6.0+cu124, NumPy 1.26.4, SciPy 1.13.1 and pandas 2.2.3, with TF32
and mixed precision disabled. Software checks and the isolated short smoke are
not scientific outcomes. Use a new persistent output directory; do not edit the
frozen scientific configuration to obtain a preferred result.

## Original archive reanalysis

The source fingerprint includes README.md and LAUNCH_AND_SHARE.md. This anonymized
copy therefore has a different documentation fingerprint from the completed
original run. Reanalyse that run using its exact code/ directory extracted from
the preserved full result archive, not this documentation-updated copy. Do not
alter or bypass its RUN_LOCK.json. Fresh runs of this package lock their own source
fingerprint normally. Root REPRODUCE.md gives both workflows and archive hashes.
