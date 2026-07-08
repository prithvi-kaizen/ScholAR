# Visual Grounding: Caption-Only vs. Full-Vision Ablation

Model: `gemma4:12b`. Same 18 cases, same prompt template and retrieved figure, differing only in whether the figure image is sent to the model.

| Condition | Mean Answer Score |
|---|---:|
| Caption + context only (no image) | 0.752 |
| Full vision (image + caption + context) | 0.731 |
