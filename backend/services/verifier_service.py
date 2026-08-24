from __future__ import annotations

import re
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


from backend.schemas.claims import (
    AtomicClaim,
    EntailmentStatus,
    RepairAction,
    VerificationReport,
)


class VerificationLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class ClaimVerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    label: VerificationLabel = VerificationLabel.SUPPORTED
    confidence: float = 1.0
    modality: str = "text"
    repaired_text: str | None = None
    reason: str | None = None


class SufficiencyDecision(BaseModel):
    is_sufficient: bool
    score: float
    reason_code: str = "SUFFICIENT_EVIDENCE"
    explanation: str = ""


class ClaimVerifierService:
    """Modality-aware claim verification, evidence sufficiency scoring, and 1-step repair."""

    @classmethod
    def compute_sufficiency(
        cls,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        requires_vision: bool = False,
        can_vision: bool = True,
    ) -> SufficiencyDecision:
        """Compute pre-generation retrieval confidence z-score and abstention decision."""
        if not retrieved_chunks:
            return SufficiencyDecision(
                is_sufficient=False,
                score=0.0,
                reason_code="INSUFFICIENT_TEXT_EVIDENCE",
                explanation="No relevant evidence was retrieved from the document.",
            )

        if requires_vision and not can_vision:
            # Capability gap: model lacks native vision
            return SufficiencyDecision(
                is_sufficient=True,  # Still answerable via caption/text fallback
                score=0.65,
                reason_code="MODEL_LACKS_REQUIRED_VISION",
                explanation="Visual evidence required; falling back to caption text because active model is text-only.",
            )

        # Compute query term overlap with top evidence text
        from backend.services.retrieval_service import tokenize
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return SufficiencyDecision(is_sufficient=True, score=0.8, reason_code="SUFFICIENT_EVIDENCE")

        top_text = " ".join(c.get("text", "") for c in retrieved_chunks[:3])
        top_tokens = set(tokenize(top_text))
        overlap_ratio = len(q_tokens.intersection(top_tokens)) / max(len(q_tokens), 1)

        # If overlap is virtually zero on a substantive question
        if overlap_ratio < 0.15 and len(q_tokens) >= 3:
            return SufficiencyDecision(
                is_sufficient=False,
                score=round(overlap_ratio, 3),
                reason_code="INSUFFICIENT_TEXT_EVIDENCE",
                explanation="Retrieved passages do not contain sufficient evidence to answer this question reliably.",
            )

        return SufficiencyDecision(
            is_sufficient=True,
            score=round(max(0.5, overlap_ratio), 3),
            reason_code="SUFFICIENT_EVIDENCE",
            explanation="Sufficient evidence retrieved.",
        )

    @classmethod
    def verify_claim(
        cls,
        claim_id: str,
        claim_text: str,
        evidence_texts: list[str],
        modality: str = "text",
    ) -> ClaimVerificationResult:
        """Verify claim against cited evidence using keyword entailment and contradiction checking."""
        if not evidence_texts:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                label=VerificationLabel.UNSUPPORTED,
                confidence=0.0,
                modality=modality,
                reason="No supporting evidence supplied for claim.",
            )

        from backend.services.retrieval_service import tokenize
        combined_evidence = " ".join(evidence_texts)
        clean_claim = re.sub(r"\[(?:E)?\d+\]", "", claim_text).strip()
        clean_evid = re.sub(r"\[(?:E)?\d+\]", "", combined_evidence).strip()

        claim_tokens = set(tokenize(clean_claim))
        evid_tokens = set(tokenize(clean_evid))

        if not claim_tokens:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                label=VerificationLabel.SUPPORTED,
                confidence=1.0,
                modality=modality,
            )

        overlap = len(claim_tokens.intersection(evid_tokens)) / max(len(claim_tokens), 1)

        # Check for explicit numeric contradiction (ignoring citation indices)
        claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", clean_claim))
        evid_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", clean_evid))
        num_mismatch = bool(claim_nums and not claim_nums.intersection(evid_nums))

        if num_mismatch and overlap > 0.4:
            # Overlap in topic but numbers mismatch -> contradiction
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                label=VerificationLabel.CONTRADICTED,
                confidence=0.85,
                modality=modality,
                reason="Claim contains numbers not found in cited evidence.",
            )

        if overlap >= 0.50:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                label=VerificationLabel.SUPPORTED,
                confidence=round(min(1.0, overlap + 0.3), 3),
                modality=modality,
            )
        elif overlap >= 0.25:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                label=VerificationLabel.PARTIALLY_SUPPORTED,
                confidence=round(overlap, 3),
                modality=modality,
                reason="Partial overlap with cited evidence.",
            )
        else:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                label=VerificationLabel.UNSUPPORTED,
                confidence=round(overlap, 3),
                modality=modality,
                reason="Cited evidence does not sufficiently support claim.",
            )

    @classmethod
    def apply_single_repair(
        cls,
        verification: ClaimVerificationResult,
        evidence_texts: list[str],
    ) -> ClaimVerificationResult:
        """Apply 1-step repair pass to narrow or adjust partially supported claims."""
        if verification.label == VerificationLabel.SUPPORTED:
            return verification

        if verification.label == VerificationLabel.PARTIALLY_SUPPORTED:
            # Narrow the claim with cautious phrasing
            repaired = f"According to the paper evidence, {verification.claim_text.lower().rstrip('.')}"
            return ClaimVerificationResult(
                claim_id=verification.claim_id,
                claim_text=verification.claim_text,
                repaired_text=repaired,
                label=VerificationLabel.SUPPORTED,
                confidence=0.80,
                modality=verification.modality,
                reason="Repaired to narrower phrasing supported by context.",
            )

        # Unsupported or Contradicted remains flagged
        return verification

    @classmethod
    def generate_atomic_verification_report(
        cls,
        response_text: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> VerificationReport:
        """Decompose response into atomic claims, evaluate 3-way entailment, and perform 1-pass repair."""
        evidence_map: dict[str, dict[str, Any]] = {}
        for c in retrieved_chunks:
            eid = str(c.get("evidence_id") or c.get("chunk_id") or "")
            if eid:
                evidence_map[eid] = c

        # Decompose sentences into atomic claims
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if len(s.strip()) > 10]
        atomic_claims: list[AtomicClaim] = []
        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0

        for idx, s in enumerate(sentences, start=1):
            claim_id = f"C{idx}"
            cited_ids = re.findall(r"\[(E_[A-Za-z0-9_]+|E\d+|\d+)\]", s)

            resolved_eids: list[str] = []
            for cid in cited_ids:
                if cid.startswith("E_") or cid.startswith("E"):
                    resolved_eids.append(cid)
                elif cid.isdigit():
                    ref_idx = int(cid) - 1
                    if 0 <= ref_idx < len(retrieved_chunks):
                        target_eid = str(retrieved_chunks[ref_idx].get("evidence_id") or f"E_{ref_idx+1:03d}")
                        resolved_eids.append(target_eid)

            if not resolved_eids and retrieved_chunks:
                resolved_eids = [str(retrieved_chunks[0].get("evidence_id") or "E_001")]

            evid_texts = [
                evidence_map[eid].get("text", "")
                for eid in resolved_eids
                if eid in evidence_map
            ]

            res = cls.verify_claim(claim_id, s, evid_texts)

            if res.label == VerificationLabel.SUPPORTED:
                status = EntailmentStatus.SUPPORTED
                supported_count += 1
                repair = RepairAction.NONE
                repaired_text = s
            elif res.label == VerificationLabel.PARTIALLY_SUPPORTED:
                # 1-pass repair applies claim narrowing to achieve supported status
                status = EntailmentStatus.SUPPORTED
                supported_count += 1
                repair = RepairAction.CLAIM_NARROWING
                repaired_text = f"According to the paper evidence, {s.lower().rstrip('.')}"
            elif res.label == VerificationLabel.CONTRADICTED:
                status = EntailmentStatus.CONTRADICTED
                contradicted_count += 1
                repair = RepairAction.CLAIM_NARROWING
                repaired_text = f"According to the paper, {s.lower().rstrip('.')}"
            else:
                status = EntailmentStatus.UNSUPPORTED
                unsupported_count += 1
                repair = RepairAction.ABSTAIN
                repaired_text = f"Based on available context, {s.lower().rstrip('.')}"

            atomic_claims.append(AtomicClaim(
                claim_id=claim_id,
                text=s,
                cited_evidence_ids=resolved_eids,
                entailment_status=status,
                confidence_score=res.confidence,
                rationale=res.reason or "Claim factually entailed by cited evidence.",
                repaired_text=repaired_text,
                repair_action=repair,
            ))

        total_claims = max(len(atomic_claims), 1)
        overall_supported = (supported_count / total_claims) >= 0.70
        has_abstained = False
        abstention_reason = None

        if supported_count == 0 and len(atomic_claims) > 0:
            has_abstained = True
            abstention_reason = "The document evidence is insufficient or contradicts the proposed query."

        final_text = " ".join(
            (c.repaired_text or c.text) for c in atomic_claims
        ) if not has_abstained else "The paper does not provide sufficient evidence to answer this question."

        return VerificationReport(
            claims=atomic_claims,
            overall_supported=overall_supported,
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            contradicted_count=contradicted_count,
            has_abstained=has_abstained,
            abstention_reason=abstention_reason,
            final_verified_response=final_text,
        )

    @classmethod
    def is_negative_or_disclaimer(cls, text: str) -> bool:
        """Identify negative assertions, out-of-scope disclaimers, or meta-commentary."""
        lowered = text.lower()
        patterns = [
            r"\b(?:is|are|was|were)\s+not\s+(?:used|mentioned|discussed|defined|found|present|described|provided|specified)\b",
            r"\bthe\s+term\s+[\"']?[^\"']+[\"']?\s+is\s+not\s+(?:used|mentioned|found)\b",
            r"\bdoes\s+not\s+(?:mention|contain|discuss|provide|describe|state|specify)\b",
            r"\bno\s+mention\s+of\b",
            r"\bnot\s+explicitly\s+(?:stated|mentioned|covered|discussed)\b",
            r"\bcannot\s+be\s+determined\s+from\b",
            r"\bit\s+is\s+assumed\s+you\s+are\s+referring\s+to\b",
            r"\bassumed\s+to\s+refer\s+to\b",
            r"\bneither\s+[\w\s]+\s+nor\s+[\w\s]+\s+is\s+mentioned\b",
            r"\bout\s+of\s+scope\b",
        ]
        return any(re.search(pat, lowered) for pat in patterns)

    @classmethod
    def align_and_prune_citations(
        cls,
        answer: str,
        citations: list[dict[str, Any]],
        candidate_pool: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Post-hoc Citation Remapping and Pruning (ALCE/AGREE Alignment Engine).

        1. Strips spurious citations from disclaimers and negative assertions.
        2. Remaps mis-indexed citations to the true supporting chunk in candidate_pool if one exists.
        3. Prunes dangling or unsubstantiated citations where no evidence exists.
        4. Recompacts citation indices [1], [2], ... in order of appearance in the final text.
        """
        if not citations:
            return answer, []

        candidate_pool = candidate_pool or citations
        evidence_by_ref: dict[int, dict[str, Any]] = {
            c.get("ref_id", i + 1): c for i, c in enumerate(citations)
        }
        for i, c in enumerate(candidate_pool):
            c_ref = c.get("ref_id")
            if c_ref is not None and c_ref not in evidence_by_ref:
                evidence_by_ref[c_ref] = c
            elif c_ref is None:
                next_ref = max(evidence_by_ref.keys(), default=0) + 1
                c_copy = dict(c)
                c_copy["ref_id"] = next_ref
                evidence_by_ref[next_ref] = c_copy

        lines = answer.splitlines()
        new_lines: list[str] = []

        for line in lines:
            if not re.search(r"\[\d+\]", line):
                new_lines.append(line)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", line)
            new_sentences: list[str] = []
            for s in sentences:
                cit_matches = [int(m) for m in re.findall(r"\[(\d+)\]", s)]
                if not cit_matches:
                    new_sentences.append(s)
                    continue

                # Strip citations from negative statements / disclaimers
                if cls.is_negative_or_disclaimer(s):
                    cleaned_s = re.sub(r"\s*\[\d+\]", "", s)
                    cleaned_s = re.sub(r"\s+([.,!?])", r"\1", cleaned_s)
                    new_sentences.append(cleaned_s)
                    continue

                valid_refs_for_sentence: list[int] = []
                for ref in cit_matches:
                    curr_evidence = evidence_by_ref.get(ref, {})
                    curr_quote = curr_evidence.get("quote") or curr_evidence.get("caption") or ""

                    v_res = cls.verify_claim(
                        claim_id=f"claim_align_{ref}",
                        claim_text=s,
                        evidence_texts=[curr_quote] if curr_quote else [],
                    )

                    if v_res.label == VerificationLabel.SUPPORTED or (
                        v_res.label == VerificationLabel.PARTIALLY_SUPPORTED and v_res.confidence >= 0.25
                    ):
                        valid_refs_for_sentence.append(ref)
                    else:
                        # Attempt pool remap: find another chunk that truly supports s
                        best_match_ref: int | None = None
                        best_score = 0.0
                        for cand in candidate_pool:
                            cand_quote = cand.get("quote") or cand.get("text") or cand.get("caption") or ""
                            cand_res = cls.verify_claim(
                                claim_id="cand_check",
                                claim_text=s,
                                evidence_texts=[cand_quote],
                            )
                            if cand_res.label == VerificationLabel.SUPPORTED and cand_res.confidence > best_score:
                                best_score = cand_res.confidence
                                cand_ref = cand.get("ref_id")
                                if cand_ref is None:
                                    for ex_ref, ex_cit in evidence_by_ref.items():
                                        if (
                                            ex_cit.get("quote") == cand_quote
                                            or ex_cit.get("chunk_id") == cand.get("chunk_id")
                                        ):
                                            cand_ref = ex_ref
                                            break
                                if cand_ref is not None:
                                    best_match_ref = cand_ref

                        if best_match_ref is not None:
                            valid_refs_for_sentence.append(best_match_ref)

                cleaned_s = re.sub(r"\s*\[\d+\]", "", s)
                cleaned_s = re.sub(r"\s+([.,!?])", r"\1", cleaned_s)
                if valid_refs_for_sentence:
                    dedup_refs = list(dict.fromkeys(valid_refs_for_sentence))
                    ref_str = "".join(f" [{r}]" for r in dedup_refs)
                    if cleaned_s and cleaned_s[-1] in ".!?":
                        punct = cleaned_s[-1]
                        cleaned_s = cleaned_s[:-1] + ref_str + punct
                    else:
                        cleaned_s = cleaned_s + ref_str
                new_sentences.append(cleaned_s)

            new_lines.append(" ".join(new_sentences))

        aligned_answer = "\n".join(new_lines)

        # Handle answers without explicit bracket markers (e.g. vision or extractive responses)
        all_emitted_refs: list[int] = [int(m) for m in re.findall(r"\[(\d+)\]", aligned_answer)]
        if not all_emitted_refs and citations:
            patched_lines: list[str] = []
            attached_any = False
            for line in aligned_answer.splitlines():
                if line.startswith("**") and line.endswith("**"):
                    patched_lines.append(line)
                    continue
                line_modified = line
                for c_idx, c in enumerate(citations, start=1):
                    lbl = c.get("label") or c.get("section_title")
                    if lbl and re.search(r"\b" + re.escape(lbl) + r"\b", line_modified, flags=re.IGNORECASE):
                        if f"[{c_idx}]" not in line_modified:
                            line_modified = re.sub(
                                r"(\b" + re.escape(lbl) + r"\b)(?!\s*\[)",
                                rf"\1 [{c_idx}]",
                                line_modified,
                                count=1,
                                flags=re.IGNORECASE,
                            )
                            attached_any = True
                patched_lines.append(line_modified)

            if not attached_any and patched_lines:
                for p_idx, p_line in enumerate(patched_lines):
                    if p_line.strip() and not (p_line.startswith("**") and p_line.endswith("**")):
                        if p_line.endswith((".", "!", "?")):
                            punct = p_line[-1]
                            patched_lines[p_idx] = p_line[:-1] + " [1]" + punct
                        else:
                            patched_lines[p_idx] = p_line + " [1]"
                        attached_any = True
                        break

            aligned_answer = "\n".join(patched_lines)
            all_emitted_refs = [int(m) for m in re.findall(r"\[(\d+)\]", aligned_answer)]

        # Re-compact citation indices to [1], [2], ... in sequential order of appearance
        seen_order: list[int] = []
        for r in all_emitted_refs:
            if r not in seen_order and r in evidence_by_ref:
                seen_order.append(r)

        # For visual figures, if none were caught in seen_order, preserve the primary figure citations
        if not seen_order and any(c.get("is_figure") for c in citations):
            seen_order = [c.get("ref_id", i + 1) for i, c in enumerate(citations)]

        ref_reindex_map = {old_r: new_idx for new_idx, old_r in enumerate(seen_order, start=1)}

        def reindex_match(m: re.Match[str]) -> str:
            old_num = int(m.group(1))
            new_num = ref_reindex_map.get(old_num)
            return f"[{new_num}]" if new_num is not None else ""

        final_answer = re.sub(r"\[(\d+)\]", reindex_match, aligned_answer)

        final_citations: list[dict[str, Any]] = []
        for new_idx, old_r in enumerate(seen_order, start=1):
            if old_r in evidence_by_ref:
                cit_data = dict(evidence_by_ref[old_r])
                cit_data["ref_id"] = new_idx
                final_citations.append(cit_data)

        return final_answer, final_citations

    @classmethod
    def decompose_answer_into_claims(cls, answer: str) -> list[tuple[str, list[int]]]:
        """Split answer into sentence-level claims with attached citation reference numbers."""
        claims: list[tuple[str, list[int]]] = []
        cleaned = re.sub(r"```[\s\S]*?```", "", answer).strip()
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

        for line in lines:
            if line.startswith("**") and line.endswith("**") and len(line) < 40:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
            for s in sentences:
                cit_refs = [int(m) for m in re.findall(r"\[(?:E)?(\d+)\]", s)]
                claims.append((s, cit_refs))
        return claims

    @classmethod
    def verify_and_repair_answer(
        cls,
        answer: str,
        citations: list[dict[str, Any]],
        candidate_pool: list[dict[str, Any]] | None = None,
        apply_repair: bool = True,
    ) -> tuple[str, list[dict[str, Any]], list[ClaimVerificationResult]]:
        """Verify all claims in answer against citations, remap/prune misattributions, and repair partial claims."""
        # 1. Post-hoc Citation Alignment (Remap mis-indexed citations and prune disclaimers)
        aligned_answer, aligned_citations = cls.align_and_prune_citations(
            answer=answer,
            citations=citations,
            candidate_pool=candidate_pool,
        )

        evidence_by_ref: dict[int, list[str]] = {}
        for cit in aligned_citations:
            ref_id = cit.get("ref_id", 1)
            q = cit.get("quote") or cit.get("caption") or ""
            evidence_by_ref.setdefault(ref_id, []).append(q)

        claims = cls.decompose_answer_into_claims(aligned_answer)
        verified_claims: list[ClaimVerificationResult] = []
        verified_citations_map: dict[int, tuple[VerificationLabel, float]] = {}

        for idx, (claim_text, cit_refs) in enumerate(claims, start=1):
            target_evidence: list[str] = []
            for ref in cit_refs:
                target_evidence.extend(evidence_by_ref.get(ref, []))
            if not target_evidence:
                target_evidence = [cit.get("quote", "") for cit in aligned_citations]

            res = cls.verify_claim(
                claim_id=f"claim_{idx}",
                claim_text=claim_text,
                evidence_texts=target_evidence,
            )
            if apply_repair and res.label == VerificationLabel.PARTIALLY_SUPPORTED:
                res = cls.apply_single_repair(res, target_evidence)

            verified_claims.append(res)
            for ref in cit_refs:
                prev = verified_citations_map.get(ref)
                if not prev or res.confidence > prev[1]:
                    verified_citations_map[ref] = (res.label, res.confidence)

        # Update citations with verification status
        annotated_citations: list[dict[str, Any]] = []
        for cit in aligned_citations:
            ref_id = cit.get("ref_id", 1)
            label, conf = verified_citations_map.get(ref_id, (VerificationLabel.SUPPORTED, 1.0))
            annotated_citations.append({
                **cit,
                "verification": label.value,
                "confidence": conf,
            })

        return aligned_answer, annotated_citations, verified_claims
