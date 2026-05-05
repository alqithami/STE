# Data directory

Keep only dataset loaders, templates, manifests, and tiny synthetic examples in git.

Do **not** commit private/raw benchmark dumps, large Chatbot Arena exports, AgentBench run logs, or generated pairwise matrices. Place those under ignored paths such as:

```text
data/raw/
data/private/
data/chatbot_arena/
data/agentbench/
```

For paper results, archive the exact raw data snapshot or provide a reproducible download/standardization script.
