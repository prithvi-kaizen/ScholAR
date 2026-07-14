# ScholAR Generation-Faithfulness Matrix (automated, 4 local models)

NLI mode: semantic. Case set: diverse (100 cases).

Automated per-model faithfulness of ScholAR's generated answers. Same metric and same set as the single-model generation-faithfulness table; only the generation model changes. No human scoring is involved.

| Model | Gen-faithfulness | Contradiction rate | Citation support | S / P / U |
|---|---:|---:|---:|---:|
| `qwen3.5:9b` | 0.888 | 0.0 | 0.872 | 505 / 71 / 3 |
| `gemma4:12b` | 0.947 | 0.001 | 0.879 | 254 / 30 / 5 |
| `llama3.1:8b` | 0.719 | 0.046 | 0.941 | 80 / 3 / 2 |
| `mistral:7b` | 0.819 | 0.006 | 0.839 | 172 / 29 / 4 |
