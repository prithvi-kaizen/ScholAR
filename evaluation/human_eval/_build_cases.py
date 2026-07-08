"""
_build_cases.py
Generates evaluation/human_eval/cases.json (100 curated human-evaluation cases)
per HUMAN_EVAL_DESIGN.md Section 7. Reproducible and reviewable: the single-document
cases are transformed mechanically from the pre-verified 51-case faithfulness set;
the visual, multi-document, and hard-retrieval cases are curated here with expert
reference answers grounded in the source papers (Transformer 1706.03762,
RAG 2005.11401, LLaMA 2302.13971, and their cited works).

Run from repo root:  python3 evaluation/human_eval/_build_cases.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation"
OUT = ROOT / "evaluation" / "human_eval" / "cases.json"
DATA = ROOT / "backend" / "data" / "papers"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_page(paper_id: str, chunk_id: str) -> int | None:
    path = DATA / paper_id / "chunks.json"
    if not path.exists():
        return None
    for c in read(path):
        if c.get("chunk_id") == chunk_id:
            return c.get("page")
    return None


def must_include_from_claim(claim: str) -> list[str]:
    """Extract salient checkable items (numbers and capitalized terms) from a
    pre-verified atomic claim, to ground Coverage scoring."""
    items: list[str] = []
    # numbers (including decimals, e.g. 28.4, 1.4, 512)
    for n in re.findall(r"\b\d+(?:\.\d+)?\b", claim):
        if n not in items:
            items.append(n)
    # multiword capitalized terms and acronyms (e.g. WMT, English-to-German, Adam)
    for term in re.findall(r"\b(?:[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*)\b", claim):
        if len(term) > 2 and term not in items and term.lower() not in {"the", "a", "an"}:
            items.append(term)
    return items[:5]


# 11 vaguest of the 51 faithfulness cases dropped to leave 40 concrete ones.
DROP = {
    "attn_parsing_result", "attn_big_base_layer_count", "rag_msmarco",
    "rag_broader_impact", "rag_appendix_details", "rag_fever_result",
    "rag_qa_multiple_answers", "llama_math_eval", "llama_triviaqa",
    "llama_approach_similar", "rag_marginalization",
}


def build_single_doc() -> list[dict]:
    src = read(EVAL / "faithfulness_cases.json")
    kept = [c for c in src if c["id"] not in DROP]
    assert len(kept) == 40, f"expected 40 single-doc, got {len(kept)}"
    out = []
    for i, c in enumerate(kept, start=1):
        page = chunk_page(c["paper_id"], c["supporting_chunk_id"])
        out.append({
            "case_id": f"he_sd_{i:02d}",
            "capability": "single_doc_text",
            "paper_id": c["paper_id"],
            "secondary_paper_ids": [],
            "question": c["query"],
            "gold_answer": c["expected_claim"],
            "must_include": must_include_from_claim(c["expected_claim"]),
            "answer_locus": f"page {page}" if page else f"chunk {c['supporting_chunk_id']}",
            "difficulty": "medium",
            "notes": f"From faithfulness benchmark ({c['claim_type']}); "
                     f"supporting chunk {c['supporting_chunk_id']}.",
            "source_faithfulness_id": c["id"],
            "claim_type": c["claim_type"],
        })
    return out


# ---- Visual cases (20): answered by the 2 multimodal models only ----
# Gold answers grounded in the Transformer (1706.03762) and RAG (2005.11401) figures.
VISUAL = [
    ("Figure 1", "1706.03762", "What does Figure 1 show about the Transformer model architecture?",
     "Figure 1 shows the full Transformer architecture: a left encoder stack and a right decoder stack, each built from repeated layers of multi-head self-attention and position-wise feed-forward sub-layers, with residual connections and layer normalization, plus encoder-decoder attention in the decoder.",
     ["encoder", "decoder", "multi-head", "self-attention", "feed-forward"], "page 3"),
    ("Figure 2", "1706.03762", "What are the two attention mechanisms shown in Figure 2?",
     "Figure 2 shows Scaled Dot-Product Attention on the left and Multi-Head Attention on the right; multi-head attention runs several scaled dot-product attention operations in parallel over learned linear projections.",
     ["Scaled Dot-Product Attention", "Multi-Head Attention"], "page 4"),
    ("Table 2", "1706.03762", "According to Table 2, what BLEU score does the Transformer (big) model achieve on English-to-German?",
     "In Table 2 the Transformer (big) achieves 28.4 BLEU on WMT 2014 English-to-German, the best reported at the time, and 41.0 (41.8) BLEU on English-to-French, at lower training cost than prior models.",
     ["28.4", "BLEU", "English-to-German"], "page 8"),
    ("Table 2", "1706.03762", "How does the Transformer compare to ByteNet and Deep-Att models in Table 2?",
     "Table 2 shows the Transformer outperforms ByteNet and Deep-Att (and other prior models) in BLEU while requiring substantially less training compute (FLOPs).",
     ["outperforms", "training cost"], "page 8"),
    ("Table 1", "1706.03762", "What does Table 1 show about the maximum path length for self-attention versus recurrent layers?",
     "Table 1 compares layer types by complexity, sequential operations, and maximum path length: self-attention has O(1) maximum path length and O(1) sequential operations, whereas recurrent layers have O(n) maximum path length and O(n) sequential operations.",
     ["self-attention", "maximum path length", "O(1)", "recurrent"], "page 6"),
    ("Table 3", "1706.03762", "Based on Table 3, what happens to model quality when the number of attention heads is varied?",
     "Table 3 (row A) shows that varying the number of attention heads while keeping computation constant affects quality: single-head attention is worse than the base 8-head setting, and too many heads also degrades quality, so 8 heads is a good balance.",
     ["attention heads", "single-head", "8 heads"], "page 9"),
    ("Figure 3", "1706.03762", "What long-distance dependency behaviour is illustrated in the attention visualization figures?",
     "The attention visualization figures show individual heads attending to distant, syntactically or semantically related words, illustrating that self-attention captures long-distance dependencies and that different heads specialize in different relations.",
     ["long-distance", "attention heads"], "page 13"),
    ("Table 4", "1706.03762", "What does Table 4 show about the Transformer on English constituency parsing?",
     "Table 4 shows the Transformer generalizes to English constituency parsing, achieving results competitive with strong prior models even with limited task-specific tuning.",
     ["constituency parsing", "competitive"], "page 10"),
    ("Figure 1", "1706.03762", "Show me the diagram that illustrates the encoder-decoder structure in the Transformer. What does it depict?",
     "Figure 1 depicts the encoder-decoder structure: the encoder maps the input sequence to continuous representations via stacked self-attention and feed-forward layers, and the decoder generates the output autoregressively using masked self-attention and encoder-decoder attention.",
     ["encoder-decoder", "self-attention", "decoder"], "page 3"),
    ("Table 2", "1706.03762", "Looking at Table 2, roughly what training cost (FLOPs) does the Transformer base model require relative to competitors?",
     "Table 2 reports the Transformer base model's training cost in FLOPs as far lower than prior state-of-the-art models while matching or exceeding their BLEU, demonstrating strong efficiency.",
     ["training cost", "FLOPs", "lower"], "page 8"),
    ("Figure 1", "2005.11401", "What does Figure 1 in the RAG paper illustrate about the retrieval-augmented pipeline?",
     "Figure 1 illustrates the RAG pipeline: a query encoder and a dense retriever (DPR) select relevant Wikipedia passages from a FAISS index, and a BART generator conditions on the query plus retrieved passages to produce the output, with the retrieved documents treated as a latent variable.",
     ["retriever", "DPR", "generator", "BART", "Wikipedia"], "page 1"),
    ("Table 1", "2005.11401", "Which datasets and metrics are summarized in the open-domain QA results table of the RAG paper?",
     "The open-domain QA results table reports exact match on Natural Questions, TriviaQA, WebQuestions, and CuratedTrec, showing RAG outperforming prior extractive and closed-book baselines.",
     ["exact match", "Natural Questions", "TriviaQA"], "page 6"),
    ("Table", "2005.11401", "How does RAG-Token compare to RAG-Sequence in the results tables of the RAG paper?",
     "The results tables show RAG-Sequence and RAG-Token perform comparably overall, with RAG-Token often better on tasks benefiting from drawing different documents for different tokens, and RAG-Sequence competitive on others.",
     ["RAG-Token", "RAG-Sequence"], "page 6"),
    ("Table 2", "1706.03762", "In Table 2, what English-to-French BLEU does the Transformer (big) reach?",
     "In Table 2 the Transformer (big) reaches 41.0 (reported as 41.8 in the big configuration) BLEU on WMT 2014 English-to-French, exceeding prior single models at lower training cost.",
     ["41.0", "English-to-French", "BLEU"], "page 8"),
    ("Figure 4", "1706.03762", "What does the attention head visualization suggest about anaphora resolution?",
     "The attention visualizations suggest some heads perform anaphora-like resolution, attending from a pronoun or reference back to the noun it refers to, evidence that heads learn interpretable linguistic relations.",
     ["attention", "anaphora", "heads"], "page 13"),
    ("Table 3", "1706.03762", "According to Table 3, what happens to performance when attention key size d_k is reduced?",
     "Table 3 shows that reducing the attention key dimension d_k hurts quality, indicating that determining compatibility is non-trivial and that a more sophisticated compatibility function than dot product may help.",
     ["key size", "d_k", "quality"], "page 9"),
    ("Table 2", "1706.03762", "Which table shows the Transformer base model training cost, and what does it indicate?",
     "Table 2 shows the training cost; it indicates the base model reaches competitive BLEU at a small fraction of the FLOPs used by prior state-of-the-art systems.",
     ["Table 2", "training cost", "FLOPs"], "page 8"),
    ("Figure 5", "1706.03762", "What pattern do the attention heads in the visualization figures exhibit?",
     "The visualization figures show that different attention heads exhibit distinct, structured patterns: some attend locally, some to distant related tokens, and some to specific syntactic roles, indicating specialization across heads.",
     ["attention heads", "patterns", "specialize"], "page 14"),
    ("Figure 2", "1706.03762", "In Figure 2, how is multi-head attention constructed from scaled dot-product attention?",
     "Figure 2 shows multi-head attention linearly projects queries, keys, and values h times, applies scaled dot-product attention to each projection in parallel, concatenates the results, and projects once more.",
     ["multi-head", "scaled dot-product", "projects", "concatenate"], "page 4"),
    ("Table 3", "1706.03762", "According to Table 3, which base Transformer setting gives the best trade-off between quality and computation?",
     "Table 3 varies base-model hyperparameters and shows the chosen base configuration (8 heads, d_model 512, d_ff 2048) gives a strong quality-computation trade-off, with quality dropping when heads, key size, or model size are reduced too far.",
     ["8 heads", "512", "trade-off"], "page 9"),
]


def build_visual() -> list[dict]:
    out = []
    for i, (label, pid, q, gold, must, locus) in enumerate(VISUAL, start=1):
        out.append({
            "case_id": f"he_vis_{i:02d}",
            "capability": "visual",
            "paper_id": pid,
            "secondary_paper_ids": [],
            "question": q,
            "gold_answer": gold,
            "must_include": must,
            "answer_locus": locus,
            "difficulty": "medium",
            "notes": f"Visual grounding on {label}. Answered by the 2 multimodal models only.",
            "expected_figure_label": label,
        })
    assert len(out) == 20, f"expected 20 visual, got {len(out)}"
    return out


# ---- Multi-document cases (20): anchor + a cited secondary paper ----
# (anchor_id, secondary_id, question, gold_answer, must_include)
MULTIDOC = [
    ("1706.03762", "1508.07909", "Which paper introduced the byte-pair encoding used for the Transformer vocabulary, and what does that method do?",
     "The byte-pair encoding used for the Transformer vocabulary comes from Sennrich et al. (1508.07909), which adapts BPE to segment words into subword units, letting the model represent rare and unseen words as sequences of frequent subwords.",
     ["byte-pair encoding", "subword", "Sennrich"]),
    ("1706.03762", "1412.6980", "The Transformer is trained with the Adam optimizer. What is Adam, according to the paper that introduced it?",
     "Adam (Kingma and Ba, 1412.6980) is an adaptive first-order optimizer that maintains exponential moving averages of gradients and squared gradients (first and second moments) with bias correction, giving per-parameter adaptive learning rates.",
     ["Adam", "adaptive", "moments"]),
    ("1706.03762", "1512.03385", "Which paper is cited for the residual connections used in each Transformer sub-layer, and what problem do residual connections address?",
     "Residual connections come from ResNet (He et al., 1512.03385), which introduced identity shortcut connections that ease optimization of very deep networks and mitigate the degradation and vanishing-gradient problems.",
     ["residual", "ResNet", "deep networks"]),
    ("2005.11401", "2004.04906", "Which paper introduced the dense retriever used in RAG, and how does it retrieve passages?",
     "RAG uses Dense Passage Retrieval (Karpukhin et al., 2004.04906), a dual-encoder that embeds questions and passages with BERT and retrieves by maximum inner-product search over dense vectors.",
     ["Dense Passage Retrieval", "dual-encoder", "inner product"]),
    ("2005.11401", "1910.13461", "Which paper introduced the BART model used as the RAG generator, and what is BART?",
     "The RAG generator is BART (Lewis et al., 1910.13461), a denoising sequence-to-sequence autoencoder pretrained by corrupting text and learning to reconstruct it, effective for generation tasks.",
     ["BART", "denoising", "sequence-to-sequence"]),
    ("2005.11401", "1702.08734", "Which library/paper provides the index used for maximum inner product search in RAG, and what does it do?",
     "RAG uses FAISS (Johnson et al., 1702.08734) for maximum inner-product search, a library for efficient similarity search and clustering of dense vectors at billion scale, including GPU acceleration.",
     ["FAISS", "similarity search", "dense vectors"]),
    ("2005.11401", "1705.03551", "Which paper introduced TriviaQA, one of the benchmarks RAG is evaluated on, and what kind of dataset is it?",
     "TriviaQA (Joshi et al., 1705.03551) is a large-scale reading-comprehension and open-domain QA dataset of trivia question-answer pairs with distantly supervised evidence documents.",
     ["TriviaQA", "question", "open-domain"]),
    ("2005.11401", "1901.08634", "Which paper introduced the Natural Questions benchmark used in RAG evaluation, and what does it contain?",
     "Natural Questions (Kwiatkowski et al., 1901.08634) is a QA benchmark of real anonymized Google queries paired with Wikipedia pages containing long-answer and short-answer annotations.",
     ["Natural Questions", "Wikipedia", "queries"]),
    ("1706.03762", "1409.0473", "Which earlier sequence-to-sequence paper is cited for the encoder-decoder attention idea, and what did it introduce?",
     "The encoder-decoder attention lineage traces to Bahdanau et al. (1409.0473), which introduced the attention mechanism for neural machine translation, letting the decoder softly align to and attend over source positions.",
     ["attention", "neural machine translation", "align"]),
    ("1706.03762", "1607.06450", "Which paper introduced layer normalization used in the Transformer sub-layers, and what does it normalize?",
     "Layer normalization (Ba et al., 1607.06450) normalizes the summed inputs across the features of a single training example (rather than across the batch), stabilizing training and working well for recurrent and attention models.",
     ["layer normalization", "features", "single example"]),
    # detail (anchor-only) cases
    ("1706.03762", None, "How does scaled dot-product attention work, and why is the scaling by the square root of d_k used?",
     "Scaled dot-product attention computes softmax(QK^T / sqrt(d_k)) V. The scaling by sqrt(d_k) counteracts large dot-product magnitudes at high dimensions that would push the softmax into regions with very small gradients.",
     ["softmax", "sqrt", "d_k", "gradients"]),
    ("2005.11401", None, "How does RAG-Token differ from RAG-Sequence in how it marginalizes over retrieved documents?",
     "RAG-Sequence uses one retrieved document to generate the entire answer and marginalizes over documents per output sequence, while RAG-Token can attend to a different document for each generated token, marginalizing per token.",
     ["RAG-Token", "RAG-Sequence", "marginalizes", "token"]),
    ("1706.03762", None, "What BLEU score did the Transformer base model achieve on WMT 2014 English-to-German?",
     "The Transformer base model achieves 27.3 BLEU on WMT 2014 English-to-German (the big model reaches 28.4).",
     ["27.3", "BLEU", "English-to-German"]),
    # 7 additional multi-document cases using the same secondary papers
    ("1706.03762", "1412.6980", "The Transformer sets Adam's beta2 to 0.98. What is the default beta2 recommended in the Adam paper, and why might the Transformer differ?",
     "The Adam paper (1412.6980) recommends default beta2 = 0.999; the Transformer instead uses 0.98, a slightly less smoothed second-moment estimate that improves stability under its warmup learning-rate schedule.",
     ["0.999", "beta2", "Adam"]),
    ("1706.03762", "1512.03385", "ResNet is cited for residual connections. What core empirical result did the ResNet paper report on ImageNet?",
     "The ResNet paper (1512.03385) showed that residual networks enable training of very deep models (for example 152 layers) and won ImageNet classification, dramatically reducing error compared to plain deep networks.",
     ["ResNet", "deep", "ImageNet"]),
    ("2005.11401", "2004.04906", "DPR is RAG's retriever. What retrieval baseline did the DPR paper show it outperforms, and by roughly how much?",
     "The DPR paper (2004.04906) showed dense retrieval substantially outperforms the traditional BM25 lexical baseline on open-domain QA passage retrieval, improving top-k retrieval accuracy by a large margin.",
     ["DPR", "BM25", "outperforms"]),
    ("2005.11401", "1910.13461", "BART is RAG's generator. What pretraining objective does the BART paper use?",
     "The BART paper (1910.13461) pretrains by corrupting text with an arbitrary noising function (for example token masking, deletion, and sentence permutation) and learning to reconstruct the original text.",
     ["BART", "corrupt", "reconstruct"]),
    ("1706.03762", "1508.07909", "BPE is used for the Transformer vocabulary. What problem in machine translation did the BPE paper originally target?",
     "The BPE paper (1508.07909) targeted the open-vocabulary and rare-word problem in neural machine translation, enabling translation of rare and unknown words via subword units instead of a fixed word vocabulary.",
     ["rare", "subword", "vocabulary"]),
    ("2005.11401", "1705.03551", "TriviaQA is a RAG benchmark. How is evidence provided in the TriviaQA dataset, according to its paper?",
     "The TriviaQA paper (1705.03551) provides distantly supervised evidence documents (from web and Wikipedia) for each question-answer pair, rather than gold labeled spans, making it a challenging QA setting.",
     ["TriviaQA", "distantly supervised", "evidence"]),
    ("1706.03762", "1409.0473", "The encoder-decoder attention derives from Bahdanau et al. What limitation of fixed-length encodings did that paper address?",
     "Bahdanau et al. (1409.0473) addressed the bottleneck of compressing a whole source sentence into a single fixed-length vector, introducing attention so the decoder can dynamically focus on relevant source words.",
     ["fixed-length", "attention", "source"]),
]


def build_multidoc() -> list[dict]:
    out = []
    for i, (anchor, sec, q, gold, must) in enumerate(MULTIDOC, start=1):
        out.append({
            "case_id": f"he_md_{i:02d}",
            "capability": "multi_doc",
            "paper_id": anchor,
            "secondary_paper_ids": [sec] if sec else [],
            "question": q,
            "gold_answer": gold,
            "must_include": must,
            "answer_locus": (f"cited paper {sec}" if sec else "anchor paper body"),
            "difficulty": "hard" if sec else "medium",
            "notes": ("Cross-paper: answer lives in the cited secondary paper."
                      if sec else "Detail: answerable from the anchor paper alone."),
        })
    assert len(out) == 20, f"expected 20 multi_doc, got {len(out)}"
    return out


# ---- Hard-retrieval cases (20): answer in the body, not the abstract ----
# (paper_id, question, gold_answer, must_include, locus)
HARD = [
    ("1706.03762", "What value of beta2 does the Transformer use in the Adam optimizer, and what epsilon?",
     "The Transformer uses Adam with beta1 = 0.9, beta2 = 0.98, and epsilon = 1e-9.",
     ["0.98", "beta2", "1e-9"], "training section"),
    ("1706.03762", "How many warmup steps does the Transformer learning-rate schedule use?",
     "The Transformer uses 4000 warmup steps, increasing the learning rate linearly for the first 4000 steps then decaying proportionally to the inverse square root of the step number.",
     ["4000", "warmup"], "training section"),
    ("1706.03762", "What is the dimensionality of the inner feed-forward layer (d_ff) in the base Transformer?",
     "The base Transformer uses an inner feed-forward dimensionality d_ff = 2048, with model dimension d_model = 512.",
     ["2048", "d_ff", "512"], "model section"),
    ("1706.03762", "How long did the base Transformer take to train and on what hardware?",
     "The base model trained for about 12 hours (100,000 steps) on 8 NVIDIA P100 GPUs; the big model trained for about 3.5 days.",
     ["12 hours", "8", "P100"], "training section"),
    ("1706.03762", "What label smoothing value did the Transformer use and what effect did it have?",
     "The Transformer used label smoothing of epsilon_ls = 0.1, which hurt perplexity (the model becomes less confident) but improved accuracy and BLEU.",
     ["0.1", "label smoothing", "BLEU"], "training section"),
    ("2005.11401", "How many top passages (K) does RAG retrieve for marginalization in its main experiments?",
     "RAG retrieves the top K latent documents (commonly K = 5 or 10 in experiments) and marginalizes over them when generating the answer.",
     ["top", "K", "marginalizes"], "method section"),
    ("2005.11401", "What exact-match score does RAG achieve on Natural Questions open-domain QA?",
     "RAG achieves 44.5 exact match on Natural Questions, outperforming prior extractive and closed-book baselines.",
     ["44.5", "Natural Questions", "exact match"], "results section"),
    ("2005.11401", "Which component of RAG is kept fixed during training and which are fine-tuned?",
     "During RAG training the document encoder and the document index are kept fixed, while the query encoder and the BART generator are fine-tuned jointly.",
     ["document encoder", "fixed", "query encoder", "generator"], "training section"),
    ("2005.11401", "What did the human evaluation of RAG find about factuality relative to BART on Jeopardy generation?",
     "Human evaluators judged RAG's Jeopardy question generations to be more factual and more specific than a BART-only baseline.",
     ["human", "more factual", "specific", "BART"], "results section"),
    ("2302.13971", "How many tokens was the LLaMA 65B model trained on?",
     "The larger LLaMA models (33B and 65B) were trained on 1.4 trillion tokens of publicly available data.",
     ["1.4 trillion", "tokens"], "training section"),
    ("2302.13971", "What normalization function and activation does LLaMA use, and where in the block?",
     "LLaMA applies pre-normalization with RMSNorm on the input of each transformer sub-layer and uses the SwiGLU activation in the feed-forward network.",
     ["RMSNorm", "pre-normalization", "SwiGLU"], "architecture section"),
    ("2302.13971", "What positional encoding does LLaMA use, replacing absolute positional embeddings?",
     "LLaMA uses rotary positional embeddings (RoPE) applied at each layer instead of absolute positional embeddings.",
     ["rotary", "RoPE", "positional"], "architecture section"),
    ("2302.13971", "What are the dimension, head count, and layer count of the LLaMA 65B model?",
     "LLaMA 65B has model dimension 8192, 64 attention heads, and 80 transformer layers.",
     ["8192", "64", "80"], "architecture table"),
    ("2302.13971", "What optimizer and learning-rate schedule does LLaMA use for training?",
     "LLaMA is trained with the AdamW optimizer and a cosine learning-rate schedule that decays the final learning rate to 10 percent of the peak, with weight decay 0.1 and gradient clipping 1.0.",
     ["AdamW", "cosine", "weight decay"], "training section"),
    ("2302.13971", "What HellaSwag zero-shot score does LLaMA 65B report?",
     "LLaMA 65B reports 84.2 on the HellaSwag commonsense reasoning benchmark in the zero-shot setting.",
     ["84.2", "HellaSwag"], "results section"),
    ("2302.13971", "Which specific public datasets make up the LLaMA pretraining mixture?",
     "The LLaMA pretraining mixture is English CommonCrawl, C4, GitHub, Wikipedia, the Books corpora (Gutenberg and Books3), ArXiv, and Stack Exchange.",
     ["CommonCrawl", "C4", "GitHub", "Wikipedia", "ArXiv"], "data section"),
    ("2302.13971", "What does the LLaMA paper report about the carbon footprint of training?",
     "The LLaMA paper reports training GPU-hours, power consumption, and estimated carbon-dioxide-equivalent emissions for its models, quantifying the environmental cost of training.",
     ["carbon", "GPU-hours", "emissions"], "environmental section"),
    ("2302.13971", "What tokenizer does LLaMA use and how are numbers handled?",
     "LLaMA uses a byte-pair encoding tokenizer from SentencePiece, splitting all numbers into individual digits and falling back to bytes for unknown UTF-8 characters.",
     ["SentencePiece", "byte-pair", "digits"], "tokenizer section"),
    ("1706.03762", "What is the per-layer computational complexity of self-attention versus a recurrent layer?",
     "Self-attention has O(n^2 * d) complexity per layer with O(1) sequential operations, while a recurrent layer has O(n * d^2) complexity with O(n) sequential operations.",
     ["n^2", "self-attention", "recurrent", "sequential"], "complexity table"),
    ("2005.11401", "What generative backbone size (parameter count) does RAG's BART generator use?",
     "RAG uses BART-large (around 400 million parameters) as its generative component.",
     ["BART-large", "400 million"], "method section"),
]


def build_hard() -> list[dict]:
    out = []
    for i, (pid, q, gold, must, locus) in enumerate(HARD, start=1):
        out.append({
            "case_id": f"he_hard_{i:02d}",
            "capability": "hard_retrieval",
            "paper_id": pid,
            "secondary_paper_ids": [],
            "question": q,
            "gold_answer": gold,
            "must_include": must,
            "answer_locus": locus,
            "difficulty": "hard",
            "notes": "Answer is in the paper body, not the abstract; single-source and specific (LitQA2-style design principle).",
        })
    assert len(out) == 20, f"expected 20 hard, got {len(out)}"
    return out


def main() -> None:
    cases = build_single_doc() + build_visual() + build_multidoc() + build_hard()
    assert len(cases) == 100, f"expected 100 total, got {len(cases)}"
    # sanity: every paper_id (and secondary) must exist on disk
    missing = set()
    for c in cases:
        for pid in [c["paper_id"], *c["secondary_paper_ids"]]:
            if not (DATA / pid).exists():
                missing.add(pid)
    if missing:
        raise SystemExit(f"Missing papers on disk: {sorted(missing)}")
    OUT.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    dist = Counter(c["capability"] for c in cases)
    print(f"Wrote {OUT} ({len(cases)} cases)")
    print("Distribution:", dict(dist))


if __name__ == "__main__":
    main()
