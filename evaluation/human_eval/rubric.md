# ScholAR Human Evaluation: Evaluator Guideline

Thank you for evaluating ScholAR. This guideline tells you exactly what to judge and how to score it. It follows the evaluation practice of comparable scientific-QA systems (SciRAG, OpenScholar, PaperQA2). Please read it once before you start, and keep it open while you score.

## What you are evaluating

ScholAR is a system that answers questions about a scientific paper and attaches citations to its statements. Each citation points to a specific page and quotes the exact sentence it relies on.

For each question, the same ScholAR pipeline was run with several different underlying language models. You will see the answers from those models, one at a time, in a random order, labeled Answer A, Answer B, and so on. You do not know which model produced which answer, and that is intentional. Score each answer on its own merits.

For each answer you will provide four quality ratings (Q1 to Q4), grade every citation in the answer (Q5), note any missing citations (Q6), and once you have seen all answers to a question, rank them (Q7).

You are given, for each question, a reference answer and a list of required facts (the "must include" list). These come from the paper and are your ground truth for judging coverage and faithfulness. When in doubt, open the cited page and read the quoted evidence yourself.

## The four quality ratings (Q1 to Q4), each scored 1 to 5

Use the whole scale. Anchors are given for 1, 3, and 5; use 2 and 4 for in-between cases.

### Q1. Relevance: does the answer stay on topic and directly address the question?

- 1: Does not address the question, or is off-topic.
- 3: Partially addresses the question but drifts or omits the core ask.
- 5: Stays tightly focused on exactly what was asked, with appropriate depth.

### Q2. Coverage: does the answer capture the key facts the question requires?

Use the reference answer and the "must include" list.

- 1: Misses most of the required facts.
- 3: Captures some required facts but omits at least one important item.
- 5: Captures all the required facts.

### Q3. Faithfulness: is every statement supported by the paper, with no fabrication or overstatement?

This is the most important rating. Judge whether the content is true to the paper, independent of how well written it is.

- 1: Contains clear fabrication, or a claim that contradicts the paper.
- 3: Mostly grounded, but includes at least one unsupported or overstated claim.
- 5: Every statement is directly supported by the paper, with no fabrication or overstatement.

### Q4. Usefulness: overall, how useful is this answer for understanding the paper?

- 1: Confusing or unhelpful; would not help a researcher.
- 3: Somewhat helpful but incomplete or hard to follow.
- 5: Clear, complete, and genuinely helpful for understanding the paper.

## Citation grading (Q5 and Q6)

Citations appear in the answer as bracketed numbers such as [1], [2]. Each one is shown with its page number and the exact sentence it quotes from the paper.

### Q5. For each citation, label whether it supports the claim it is attached to.

Read the claim in the answer that carries the citation, then read the quoted evidence, and choose:

- Supported: the cited evidence directly and fully supports the claim.
- Partial: the cited evidence is related and supports the claim in part, but does not fully establish it. For example it backs one part of a multi-part claim, or it is on the right topic but does not contain the specific number or detail the claim states.
- Unsupported: the cited evidence does not support the claim. For example it is on the wrong page or figure, is topically unrelated, or contradicts the claim. This is a citation error.

### Q6. Missing citations.

List any statement in the answer that makes a specific, checkable, paper-specific factual claim but carries no citation. If there are none, leave it blank. A general framing sentence does not need a citation; a specific factual assertion does.

## Worked citation examples

Suppose the question is "What BLEU score does the Transformer achieve on English-to-German?"

- Answer says: "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German [1]." Citation [1] quotes: "our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task." Label: Supported. The evidence contains the exact claim.
- Answer says: "The Transformer achieves 28.4 BLEU on English-to-German [1]." Citation [1] quotes: "we evaluate on the WMT 2014 English-to-German translation task." Label: Partial. The evidence confirms the task but not the specific 28.4 number.
- Answer says: "The Transformer achieves 28.4 BLEU on English-to-German [1]." Citation [1] quotes: "dropout is applied to the output of each sub-layer." Label: Unsupported. The evidence is unrelated to the claim.

## Comparative ranking (Q7)

After you have scored all the answers to one question, rank them from best to worst overall, considering all four ratings together with a focus on faithfulness. Ties are allowed if two answers are genuinely equal.

## Protocol notes

- Score each answer independently. Do not let a strong answer to a question change how you score a weaker one, except in the final ranking.
- Judge faithfulness against the paper, not against your own prior knowledge. If a statement is true in general but not stated in this paper, and the answer presents it as this paper's content, that lowers faithfulness.
- A subset of questions is scored by more than one evaluator so we can measure agreement. Please score every question you are assigned, including any that look similar to ones you have already seen.
- Do not try to identify which model wrote which answer. The labels A, B, C, D are randomized per question.

## What we compute from your scores

Your ratings produce, per model: the mean of each quality dimension, the fraction of citations that are Supported, and citation precision and recall. Because the same pipeline is run across several models, comparing these per-model numbers tells us whether ScholAR's grounding holds up regardless of the underlying model. Your faithfulness ratings are also compared against the system's automated faithfulness metric to check that the automated metric agrees with expert judgment.
