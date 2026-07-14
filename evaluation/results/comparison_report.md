# ScholAR vs Local Baselines (same model, same cases)

Citation quality is ALCE/OpenScholar-style: precision = supported / cited citations, recall = fraction of factual sentences that carry a citation.

| System | N | Gen-faith | Cite-P | Cite-R | Cite-F1 | Must-incl | Gold-cite-R | Invalid-page |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pdfchat | 91 | 0.801 | 0.915 | 0.699 | 0.742 | 0.267 | 0.255 | 0.0 |
| vanilla_rag | 91 | 0.86 | 0.901 | 0.776 | 0.802 | 0.34 | 0.49 | 0.0 |
| paperqa2 | 91 | 0.872 | 0.824 | 0.793 | 0.782 | 0.55 | 0.784 | 0.0 |
| scholar | 91 | 0.892 | 0.85 | 0.739 | 0.76 | 0.563 | 0.706 | 0.0 |
| scholar_rcs | 1 | 0.6 | 1.0 | 0.667 | 0.8 | 0.0 | None | 0.0 |
