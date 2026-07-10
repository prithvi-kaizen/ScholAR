# ScholAR Generation-Faithfulness Report

Model: `gemma4:12b`. NLI mode: semantic. Cases: 51.

This measures the faithfulness of ScholAR's **generated answers**, not of a pre-written gold claim. It answers: does the generated answer stay grounded in the retrieved context, and are its inline citations actually supporting? Contrast with the retrieval-support CFS in `faithfulness_eval_report_v3.md`, which scores gold claims against retrieval.

| Metric | Value |
|---|---:|
| Mean generation-faithfulness (answer atoms entailed) | 0.971 |
| Mean hallucination rate (answer atoms contradicted) | 0.0 |
| Citation-support rate (Supported / all checked) | 0.94 |
| Citations checked (Supported / Partial / Unsupported) | 188 / 12 / 0 |

## Generation faithfulness by claim type

| Claim type | N | Mean generation-faithfulness |
|---|---:|---:|
| architecture_detail | 10 | 1.0 |
| conceptual_claim | 5 | 0.933 |
| environmental_claim | 1 | 1.0 |
| formula | 1 | 1.0 |
| human_eval | 2 | 1.0 |
| result_number | 13 | 0.952 |
| technical_claim | 11 | 0.97 |
| training_detail | 8 | 0.975 |
