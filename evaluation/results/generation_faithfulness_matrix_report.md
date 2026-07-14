# ScholAR Generation-Faithfulness Matrix (automated, 4 local models)

NLI mode: semantic. Case set: diverse (100 cases).

Automated per-model faithfulness of ScholAR's generated answers. Same metric and same set as the single-model generation-faithfulness table; only the generation model changes. No human scoring is involved.

| Model | Gen-faithfulness | Contradiction rate | Citation support | S / P / U |
|---|---:|---:|---:|---:|
| `qwen3.5:9b` | 0.908 | 0.003 | 0.888 | 521 / 61 / 5 |
| `gemma4:12b` | 0.951 | 0.001 | 0.907 | 332 / 31 / 3 |
| `llama3.1:8b` | 0.719 | 0.041 | 0.951 | 77 / 3 / 1 |
| `mistral:7b` | 0.809 | 0.006 | 0.868 | 198 / 25 / 5 |
