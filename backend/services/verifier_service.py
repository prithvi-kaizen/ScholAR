from __future__ import annotations

import re
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from backend.schemas.answer_trace import InterventionControls, RepairMode
from backend.schemas.claims import (
    AtomicClaim,
    CitationSpan,
    ClaimRepairRecord,
    EntailmentStatus,
    EvidenceProvenance,
    RepairAction,
    VerificationReport,
)


_CITATION_PATTERN = re.compile(
    r"\[\s*(?:E_?[A-Za-z0-9_]+|\d+)(?:\s*(?:,|;|/|&|and)\s*(?:E_?[A-Za-z0-9_]+|\d+))*\s*\]",
    flags=re.IGNORECASE,
)
_REFERENCE_PATTERN = re.compile(r"E_[A-Za-z0-9_]+|E\d+|\d+", flags=re.IGNORECASE)
_ABSTENTION_TEXT = "The paper does not provide sufficient evidence to answer this question."


class VerificationLabel(str, Enum):
    """Canonical verifier labels; the long partial name remains a code alias only."""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    PARTIALLY_SUPPORTED = "PARTIAL"
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
    start: int | None = None
    end: int | None = None
    repair_action: RepairAction = RepairAction.NONE


class SufficiencyDecision(BaseModel):
    is_sufficient: bool
    score: float
    reason_code: str = "SUFFICIENT_EVIDENCE"
    explanation: str = ""


class ParsedCitation(BaseModel):
    start: int
    end: int
    marker: str
    reference_ids: list[str] = Field(default_factory=list)


class ParsedClaim(BaseModel):
    claim_id: str
    text: str
    start: int
    end: int
    normalized_text: str
    claim_type: str = "factual"
    citations: list[ParsedCitation] = Field(default_factory=list)


class VerificationOutcome(BaseModel):
    original_answer: str
    final_answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    initial_results: list[ClaimVerificationResult] = Field(default_factory=list)
    final_results: list[ClaimVerificationResult] = Field(default_factory=list)
    initial_report: VerificationReport
    final_report: VerificationReport
    edits: list[ClaimRepairRecord] = Field(default_factory=list)
    reverified: bool = False


class SupportScorer(Protocol):
    """Provider-neutral support scorer interface used by the repair policy."""

    backend: str
    version: str

    def score(
        self,
        claim_id: str,
        claim_text: str,
        evidence_texts: list[str],
        modality: str,
        cited_evidence_ids: list[str],
    ) -> ClaimVerificationResult: ...


class LexicalSupportScorer:
    """Traceable lexical baseline. Thresholds are intentionally not called calibrated."""

    backend = "lexical-overlap"
    version = "lexical-support-v2"
    supported_threshold = 0.50
    partial_threshold = 0.25

    def score(
        self,
        claim_id: str,
        claim_text: str,
        evidence_texts: list[str],
        modality: str,
        cited_evidence_ids: list[str],
    ) -> ClaimVerificationResult:
        if not evidence_texts or not any(text.strip() for text in evidence_texts):
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                cited_evidence_ids=cited_evidence_ids,
                label=VerificationLabel.UNSUPPORTED,
                confidence=0.0,
                modality=modality,
                reason="No cited evidence was supplied for the claim.",
            )

        from backend.services.retrieval_service import tokenize

        combined_evidence = " ".join(evidence_texts)
        clean_claim = _CITATION_PATTERN.sub("", claim_text).strip()
        clean_evidence = _CITATION_PATTERN.sub("", combined_evidence).strip()
        claim_tokens = set(tokenize(clean_claim))
        evidence_tokens = set(tokenize(clean_evidence))

        if not claim_tokens:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                cited_evidence_ids=cited_evidence_ids,
                label=VerificationLabel.SUPPORTED,
                confidence=1.0,
                modality=modality,
            )

        overlap = len(claim_tokens.intersection(evidence_tokens)) / max(len(claim_tokens), 1)

        # Negation / Polar contradiction detection
        _NEGATION_TERMS = frozenset({
            "not", "no", "never", "neither", "nor", "fails", "fail", "failed",
            "doesnt", "doesn't", "isnt", "isn't", "wont", "won't", "without",
            "cannot", "can't", "deteriorates", "degrades",
        })
        claim_words = set(re.findall(r"\b[a-z']+\b", clean_claim.lower()))
        evidence_words = set(re.findall(r"\b[a-z']+\b", clean_evidence.lower()))

        claim_neg = claim_words.intersection(_NEGATION_TERMS)
        evidence_neg = evidence_words.intersection(_NEGATION_TERMS)
        has_polar_asymmetry = bool(claim_neg) != bool(evidence_neg)

        content_claim = claim_words - _NEGATION_TERMS
        content_evidence = evidence_words - _NEGATION_TERMS
        content_overlap = (
            len(content_claim.intersection(content_evidence)) / max(len(content_claim), 1)
            if content_claim else 0.0
        )

        if has_polar_asymmetry and content_overlap >= 0.45:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                cited_evidence_ids=cited_evidence_ids,
                label=VerificationLabel.CONTRADICTED,
                confidence=0.95,
                modality=modality,
                reason="Polar negation contradiction: claim polarity directly opposes the cited evidence.",
            )

        claim_without_structural_numbers = re.sub(
            r"\b(?:figure|table|section|page|eq|equation|model|step|ref|row|column)\s+\d+\b",
            "",
            clean_claim,
            flags=re.IGNORECASE,
        )
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_without_structural_numbers))
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", clean_evidence))

        unsupported_claim_numbers = claim_numbers - evidence_numbers
        if unsupported_claim_numbers and overlap >= 0.35:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                cited_evidence_ids=cited_evidence_ids,
                label=VerificationLabel.CONTRADICTED,
                confidence=0.95,
                modality=modality,
                reason=f"Numerical contradiction: value(s) {sorted(list(unsupported_claim_numbers))} in claim are not supported by the cited evidence.",
            )

        if overlap >= self.supported_threshold:
            label = VerificationLabel.SUPPORTED
            reason = "Claim tokens are supported by the cited evidence under the lexical baseline."
        elif overlap >= self.partial_threshold:
            label = VerificationLabel.PARTIAL
            reason = "Only part of the claim overlaps the cited evidence."
        else:
            label = VerificationLabel.UNSUPPORTED
            reason = "Cited evidence does not contain enough of the claim to support it."
        return ClaimVerificationResult(
            claim_id=claim_id,
            claim_text=claim_text,
            cited_evidence_ids=cited_evidence_ids,
            label=label,
            confidence=round(overlap, 3),
            modality=modality,
            reason=reason,
        )


class ClaimVerifierService:
    """Span-preserving claim verification with deterministic selective repair."""

    scorer: SupportScorer = LexicalSupportScorer()

    @classmethod
    def compute_sufficiency(
        cls,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        requires_vision: bool = False,
        can_vision: bool = True,
    ) -> SufficiencyDecision:
        """Compute pre-generation retrieval overlap and an auditable abstention decision."""
        if not retrieved_chunks:
            return SufficiencyDecision(
                is_sufficient=False,
                score=0.0,
                reason_code="INSUFFICIENT_TEXT_EVIDENCE",
                explanation="No relevant evidence was retrieved from the document.",
            )
        ranked_visual_candidates = [
            chunk
            for chunk in retrieved_chunks
            if chunk.get("is_figure_chunk")
            and (
                (
                    chunk.get("image_embedding_eligible") is True
                    and type(chunk.get("image_embedding_rank")) is int
                    and chunk["image_embedding_rank"] <= 3
                    and isinstance(chunk.get("image_embedding_score"), (int, float))
                    and not isinstance(chunk.get("image_embedding_score"), bool)
                    and isinstance(chunk.get("image_embedding_threshold"), (int, float))
                    and not isinstance(chunk.get("image_embedding_threshold"), bool)
                    and float(chunk["image_embedding_score"])
                    >= float(chunk["image_embedding_threshold"])
                )
                or (
                    chunk.get("page_image_eligible") is True
                    and type(chunk.get("page_image_rank")) is int
                    and chunk["page_image_rank"] <= 3
                    and isinstance(chunk.get("page_image_score"), (int, float))
                    and not isinstance(chunk.get("page_image_score"), bool)
                    and isinstance(chunk.get("page_image_threshold"), (int, float))
                    and not isinstance(chunk.get("page_image_threshold"), bool)
                    and float(chunk["page_image_score"])
                    >= float(chunk["page_image_threshold"])
                )
            )
        ]

        from backend.services.retrieval_service import tokenize

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return SufficiencyDecision(is_sufficient=True, score=0.8)
        top_tokens = set(tokenize(" ".join(str(c.get("text", "")) for c in retrieved_chunks[:3])))
        overlap = len(query_tokens.intersection(top_tokens)) / max(len(query_tokens), 1)
        if overlap < 0.15 and len(query_tokens) >= 3:
            if requires_vision and not can_vision:
                return SufficiencyDecision(
                    is_sufficient=False,
                    score=round(overlap, 3),
                    reason_code="MODEL_LACKS_REQUIRED_VISION",
                    explanation=(
                        "Text/caption evidence is insufficient and the active model cannot inspect pixels."
                    ),
                )
            if can_vision and ranked_visual_candidates:
                return SufficiencyDecision(
                    is_sufficient=False,
                    score=round(overlap, 3),
                    reason_code="VISUAL_INSPECTION_REQUIRED",
                    explanation=(
                        "Text evidence is insufficient. A score-floor-qualified image is only a "
                        "candidate for pixel inspection; its uncalibrated cosine does not establish sufficiency."
                    ),
                )
            return SufficiencyDecision(
                is_sufficient=False,
                score=round(overlap, 3),
                reason_code="INSUFFICIENT_TEXT_EVIDENCE",
                explanation="Retrieved passages do not contain sufficient evidence to answer reliably.",
            )
        return SufficiencyDecision(
            is_sufficient=True,
            score=round(max(0.5, overlap), 3),
            explanation="Sufficient text or caption evidence retrieved.",
        )

    @classmethod
    def verify_claim(
        cls,
        claim_id: str,
        claim_text: str,
        evidence_texts: list[str],
        modality: str = "text",
        cited_evidence_ids: list[str] | None = None,
    ) -> ClaimVerificationResult:
        return cls.scorer.score(
            claim_id,
            claim_text,
            evidence_texts,
            modality,
            cited_evidence_ids or [],
        )

    @staticmethod
    def _sentence_ranges(line: str) -> list[tuple[int, int]]:
        """Return conservative sentence ranges without losing source offsets."""
        abbreviations = ("e.g.", "i.e.", "et al.", "fig.", "eq.", "dr.", "mr.", "mrs.", "vs.")
        ranges: list[tuple[int, int]] = []
        start = 0
        for index, char in enumerate(line):
            if char not in ".!?":
                continue
            if char == "." and index > 0 and index + 1 < len(line):
                if line[index - 1].isdigit() and line[index + 1].isdigit():
                    continue
            prefix = line[max(0, index - 8):index + 1].lower()
            if char == "." and any(prefix.endswith(abbreviation) for abbreviation in abbreviations):
                continue
            if index + 1 < len(line) and not line[index + 1].isspace():
                continue
            ranges.append((start, index + 1))
            start = index + 1
        if start < len(line):
            ranges.append((start, len(line)))
        return ranges

    @classmethod
    def parse_answer_into_claims(cls, answer: str) -> list[ParsedClaim]:
        """Parse claims while preserving exact zero-based, half-open answer offsets."""
        claims: list[ParsedClaim] = []
        in_fence = False
        for line_match in re.finditer(r"[^\n]*(?:\n|$)", answer):
            raw_line = line_match.group(0)
            if not raw_line:
                continue
            line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not stripped:
                continue
            if re.match(r"^\s{0,3}#{1,6}\s+", line):
                continue
            if stripped.startswith("|") and stripped.endswith("|"):
                continue
            if stripped.startswith("**") and stripped.endswith("**") and not re.search(r"[.!?]$", stripped):
                continue

            for local_start, local_end in cls._sentence_ranges(line):
                segment = line[local_start:local_end]
                leading = len(segment) - len(segment.lstrip())
                trailing = len(segment.rstrip())
                if trailing <= leading:
                    continue
                start = line_match.start() + local_start + leading
                end = line_match.start() + local_start + trailing
                text = answer[start:end]
                if not text or text in {"-", "*"}:
                    continue
                citation_matches = list(_CITATION_PATTERN.finditer(text))
                citations = [
                    ParsedCitation(
                        start=start + match.start(),
                        end=start + match.end(),
                        marker=match.group(0),
                        reference_ids=[ref.upper() for ref in _REFERENCE_PATTERN.findall(match.group(0))],
                    )
                    for match in citation_matches
                ]
                normalized = _CITATION_PATTERN.sub("", text)
                normalized = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", normalized).strip()
                claim_type = "disclaimer" if cls.is_negative_or_disclaimer(normalized) else "factual"
                claims.append(ParsedClaim(
                    claim_id=f"C{len(claims) + 1}",
                    text=text,
                    start=start,
                    end=end,
                    normalized_text=normalized,
                    claim_type=claim_type,
                    citations=citations,
                ))
        return claims

    @classmethod
    def decompose_answer_into_claims(cls, answer: str) -> list[tuple[str, list[int]]]:
        """Compatibility wrapper around the span-preserving parser."""
        decomposed: list[tuple[str, list[int]]] = []
        for claim in cls.parse_answer_into_claims(answer):
            if claim.claim_type != "factual":
                continue
            refs = [
                int(ref[1:] if ref.startswith("E") and ref[1:].isdigit() else ref)
                for citation in claim.citations
                for ref in citation.reference_ids
                if ref.isdigit() or (ref.startswith("E") and ref[1:].isdigit())
            ]
            decomposed.append((claim.text, refs))
        return decomposed

    @staticmethod
    def _evidence_text(item: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("label", "caption", "section_title", "quote", "visual_observation", "body_text", "text"):
            val = item.get(key)
            if isinstance(val, str) and val.strip() and val.strip() not in parts:
                parts.append(val.strip())
        return " ".join(parts) if parts else ""

    @staticmethod
    def _evidence_id(item: dict[str, Any], fallback: str) -> str:
        return str(
            item.get("evidence_id")
            or item.get("source_evidence_id")
            or item.get("chunk_id")
            or fallback
        )

    @classmethod
    def _prepare_evidence(
        cls,
        citations: list[dict[str, Any]],
        candidate_pool: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        used_refs: set[int] = set()

        def append(item: dict[str, Any], fallback_index: int) -> None:
            text = cls._evidence_text(item)
            evidence_id = cls._evidence_id(item, f"E_{fallback_index:03d}")
            identity = (evidence_id, text)
            if identity in seen:
                return
            copy = dict(item)
            ref = copy.get("ref_id")
            try:
                ref_id = int(ref) if ref is not None else None
            except (TypeError, ValueError):
                ref_id = None
            if ref_id is None or ref_id in used_refs:
                ref_id = max(used_refs, default=0) + 1
                while ref_id in used_refs:
                    ref_id += 1
            copy["ref_id"] = ref_id
            copy["_evidence_id"] = evidence_id
            copy["_evidence_text"] = text
            records.append(copy)
            seen.add(identity)
            used_refs.add(ref_id)

        for index, citation in enumerate(citations, start=1):
            append(citation, index)
        for index, candidate in enumerate(candidate_pool or [], start=len(records) + 1):
            append(candidate, index)

        by_ref = {int(record["ref_id"]): record for record in records}
        by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            evidence_id = str(record["_evidence_id"])
            aliases = {
                evidence_id.upper(),
                str(record.get("evidence_id") or "").upper(),
                str(record.get("source_evidence_id") or "").upper(),
                str(record.get("chunk_id") or "").upper(),
                f"E{record['ref_id']}",
            }
            for alias in aliases:
                if alias:
                    by_id[alias] = record
        return records, by_ref, by_id

    @classmethod
    def _resolve_claim_evidence(
        cls,
        claim: ParsedClaim,
        by_ref: dict[int, dict[str, Any]],
        by_id: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        seen: set[int] = set()
        for marker in claim.citations:
            for raw_ref in marker.reference_ids:
                record: dict[str, Any] | None = None
                if raw_ref.isdigit():
                    record = by_ref.get(int(raw_ref))
                else:
                    record = by_id.get(raw_ref.upper())
                if record is not None and id(record) not in seen:
                    seen.add(id(record))
                    resolved.append(record)
        return resolved

    @staticmethod
    def _status(label: VerificationLabel) -> EntailmentStatus:
        return {
            VerificationLabel.SUPPORTED: EntailmentStatus.SUPPORTED,
            VerificationLabel.PARTIAL: EntailmentStatus.PARTIAL,
            VerificationLabel.UNSUPPORTED: EntailmentStatus.UNSUPPORTED,
            VerificationLabel.CONTRADICTED: EntailmentStatus.CONTRADICTED,
        }[label]

    @classmethod
    def _to_atomic_claim(
        cls,
        claim: ParsedClaim,
        result: ClaimVerificationResult,
        evidence: list[dict[str, Any]],
        *,
        second_pass: bool,
    ) -> AtomicClaim:
        status = cls._status(result.label)
        evidence_ids = [str(item["_evidence_id"]) for item in evidence]
        spans: list[CitationSpan] = []
        for marker in claim.citations:
            marker_evidence = cls._resolve_claim_evidence(
                ParsedClaim(
                    claim_id=claim.claim_id,
                    text=marker.marker,
                    start=marker.start,
                    end=marker.end,
                    normalized_text="",
                    citations=[marker],
                ),
                {int(item["ref_id"]): item for item in evidence},
                {str(item["_evidence_id"]).upper(): item for item in evidence},
            )
            spans.append(CitationSpan(
                start=marker.start,
                end=marker.end,
                marker=marker.marker,
                reference_ids=marker.reference_ids,
                evidence_ids=[str(item["_evidence_id"]) for item in marker_evidence],
            ))
        provenance = [
            EvidenceProvenance(
                evidence_id=str(item["_evidence_id"]),
                ref_id=int(item["ref_id"]),
                source_paper_id=item.get("source_paper_id"),
                document_id=item.get("document_id"),
                page=item.get("page"),
                region=item.get("bbox_normalized") or item.get("bbox"),
            )
            for item in evidence
        ]
        return AtomicClaim(
            claim_id=claim.claim_id,
            text=claim.text,
            cited_evidence_ids=evidence_ids,
            entailment_status=status,
            confidence_score=result.confidence,
            rationale=result.reason or "Non-factual response structure does not require evidence.",
            start=claim.start,
            end=claim.end,
            citation_spans=spans,
            normalized_text=claim.normalized_text,
            claim_type=claim.claim_type,
            resolved_evidence=provenance,
            first_pass_status=None if second_pass else status,
            second_pass_status=status if second_pass else None,
            final_start=claim.start if second_pass else None,
            final_end=claim.end if second_pass else None,
        )

    @classmethod
    def _evaluate_answer(
        cls,
        answer: str,
        by_ref: dict[int, dict[str, Any]],
        by_id: dict[str, dict[str, Any]],
        *,
        second_pass: bool,
    ) -> tuple[list[AtomicClaim], list[ClaimVerificationResult], dict[str, list[dict[str, Any]]]]:
        atomic_claims: list[AtomicClaim] = []
        results: list[ClaimVerificationResult] = []
        evidence_by_claim: dict[str, list[dict[str, Any]]] = {}
        for claim in cls.parse_answer_into_claims(answer):
            evidence = cls._resolve_claim_evidence(claim, by_ref, by_id)
            evidence_by_claim[claim.claim_id] = evidence
            if claim.claim_type != "factual":
                result = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    label=VerificationLabel.SUPPORTED,
                    confidence=1.0,
                    reason="Non-factual disclaimer or response structure.",
                )
            else:
                result = cls.verify_claim(
                    claim.claim_id,
                    claim.text,
                    [str(item["_evidence_text"]) for item in evidence],
                    cited_evidence_ids=[str(item["_evidence_id"]) for item in evidence],
                )
            result.start = claim.start
            result.end = claim.end
            results.append(result)
            atomic_claims.append(cls._to_atomic_claim(claim, result, evidence, second_pass=second_pass))
        return atomic_claims, results, evidence_by_claim

    @classmethod
    def _report(
        cls,
        answer: str,
        atomic_claims: list[AtomicClaim],
        *,
        edits: list[ClaimRepairRecord] | None = None,
        second_pass: bool = False,
        abstained: bool | None = None,
    ) -> VerificationReport:
        factual = [claim for claim in atomic_claims if claim.claim_type == "factual"]
        supported = sum(claim.entailment_status == EntailmentStatus.SUPPORTED for claim in factual)
        partial = sum(claim.entailment_status == EntailmentStatus.PARTIAL for claim in factual)
        unsupported = sum(claim.entailment_status == EntailmentStatus.UNSUPPORTED for claim in factual)
        contradicted = sum(claim.entailment_status == EntailmentStatus.CONTRADICTED for claim in factual)
        should_abstain = (bool(factual) and supported == 0) if abstained is None else abstained
        overall_supported = unsupported == 0 and contradicted == 0 and partial == 0
        return VerificationReport(
            claims=atomic_claims,
            overall_supported=overall_supported,
            supported_count=supported,
            partial_count=partial,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
            has_abstained=should_abstain,
            abstention_reason=(
                "No factual claim is supported by its cited evidence."
                if should_abstain else None
            ),
            final_verified_response=_ABSTENTION_TEXT if should_abstain and not second_pass else answer,
            edits=edits or [],
            second_pass_completed=second_pass,
        )

    @staticmethod
    def _replace_citations(claim_text: str, ref_id: int) -> str:
        marker = f"[{ref_id}]"
        if _CITATION_PATTERN.search(claim_text):
            first = True

            def replace(_: re.Match[str]) -> str:
                nonlocal first
                if first:
                    first = False
                    return marker
                return ""

            return _CITATION_PATTERN.sub(replace, claim_text)
        if claim_text.endswith((".", "!", "?")):
            return f"{claim_text[:-1].rstrip()} {marker}{claim_text[-1]}"
        return f"{claim_text.rstrip()} {marker}"

    @classmethod
    def _supported_narrowing(
        cls,
        claim: ParsedClaim,
        evidence: list[dict[str, Any]],
    ) -> tuple[str, ClaimVerificationResult] | None:
        if not evidence:
            return None
        marker_text = "".join(citation.marker for citation in claim.citations)
        without_markers = _CITATION_PATTERN.sub("", claim.text)
        terminal = without_markers[-1] if without_markers.endswith((".", "!", "?")) else ""
        body = without_markers[:-1] if terminal else without_markers
        pieces = [
            piece
            for piece in re.split(r"(?:\s*;\s*|,?\s+(?:and|but|while|whereas|although|however)\s+)", body)
            if piece.strip()
        ]
        candidates: list[tuple[str, ClaimVerificationResult]] = []
        for piece in pieces:
            retained = piece.rstrip(" ,;")
            if not retained.strip() or retained.strip() == body.strip():
                continue
            replacement = retained
            if marker_text:
                replacement = f"{replacement.rstrip()} {marker_text}"
            if terminal:
                replacement = f"{replacement.rstrip()}{terminal}"
            result = cls.verify_claim(
                claim.claim_id,
                replacement,
                [str(item["_evidence_text"]) for item in evidence],
                cited_evidence_ids=[str(item["_evidence_id"]) for item in evidence],
            )
            if result.label in {VerificationLabel.SUPPORTED, VerificationLabel.PARTIAL} and replacement != claim.text:
                candidates.append((replacement, result))
        return max(candidates, key=lambda item: len(item[0]), default=None)

    @classmethod
    def apply_single_repair(
        cls,
        verification: ClaimVerificationResult,
        evidence_texts: list[str],
    ) -> ClaimVerificationResult:
        """Compatibility helper: narrow by deletion and reverify; never support by hedging."""
        if verification.label == VerificationLabel.SUPPORTED:
            return verification
        if verification.label != VerificationLabel.PARTIAL:
            return verification
        parsed = cls.parse_answer_into_claims(verification.claim_text)
        if not parsed:
            return verification
        evidence = [
            {"ref_id": index, "_evidence_id": f"E{index}", "_evidence_text": text}
            for index, text in enumerate(evidence_texts, start=1)
        ]
        narrowed = cls._supported_narrowing(parsed[0], evidence)
        if narrowed is None:
            return verification.model_copy(update={
                "label": VerificationLabel.UNSUPPORTED,
                "repaired_text": "",
                "repair_action": RepairAction.CLAIM_DELETION,
                "reason": "No deletion-derived subclaim passed second-pass support.",
            })
        repaired_text, second_pass = narrowed
        return second_pass.model_copy(update={
            "claim_text": verification.claim_text,
            "repaired_text": repaired_text,
            "repair_action": RepairAction.CLAIM_NARROWING,
            "reason": "Unsupported clause removed; retained text passed a second support check.",
        })

    @classmethod
    def generate_atomic_verification_report(
        cls,
        response_text: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> VerificationReport:
        """Report support for an exact response without fabricating repairs or evidence."""
        _, by_ref, by_id = cls._prepare_evidence([], retrieved_chunks)
        claims, _, _ = cls._evaluate_answer(response_text, by_ref, by_id, second_pass=False)
        return cls._report(response_text, claims)

    @classmethod
    def verify_and_repair_detailed(
        cls,
        answer: str,
        citations: list[dict[str, Any]],
        candidate_pool: list[dict[str, Any]] | None = None,
        apply_repair: bool | None = None,
        controls: InterventionControls | None = None,
    ) -> VerificationOutcome:
        """Execute the requested typed intervention, then verify retained claims."""
        if controls is None:
            enabled = True if apply_repair is None else apply_repair
            controls = InterventionControls(
                repair_mode=RepairMode.SELECTIVE if enabled else RepairMode.NONE,
                abstain_on_no_supported_claims=enabled,
            )
        elif apply_repair is not None:
            raise ValueError("pass controls or apply_repair, not both")
        records, by_ref, by_id = cls._prepare_evidence(citations, candidate_pool)
        parsed_claims = cls.parse_answer_into_claims(answer)
        initial_claims, initial_results, evidence_by_claim = cls._evaluate_answer(
            answer, by_ref, by_id, second_pass=False
        )
        initial_report = cls._report(answer, initial_claims)
        if controls.repair_mode == RepairMode.NONE:
            return VerificationOutcome(
                original_answer=answer,
                final_answer=answer,
                citations=citations,
                initial_results=initial_results,
                final_results=initial_results,
                initial_report=initial_report,
                final_report=initial_report,
                reverified=False,
            )

        result_by_id = {result.claim_id: result for result in initial_results}
        edits: list[ClaimRepairRecord] = []
        replacements: list[tuple[int, int, str]] = []

        for claim in parsed_claims:
            result = result_by_id[claim.claim_id]
            evidence = evidence_by_claim.get(claim.claim_id, [])
            original_evidence_ids = [str(item["_evidence_id"]) for item in evidence]

            if claim.claim_type != "factual":
                if controls.repair_mode != RepairMode.SELECTIVE:
                    continue
                for marker in claim.citations:
                    marker_start = marker.start
                    if marker_start > claim.start and answer[marker_start - 1].isspace():
                        marker_start -= 1
                    original_marker = answer[marker_start:marker.end]
                    replacements.append((marker_start, marker.end, ""))
                    edits.append(ClaimRepairRecord(
                        claim_id=claim.claim_id,
                        action=RepairAction.CITATION_REMAP,
                        original_start=marker_start,
                        original_end=marker.end,
                        original_text=original_marker,
                        replacement_text="",
                        initial_status=EntailmentStatus.SUPPORTED,
                        second_pass_status=EntailmentStatus.SUPPORTED,
                        original_evidence_ids=original_evidence_ids,
                    ))
                continue
            if result.label == VerificationLabel.SUPPORTED:
                continue

            replacement: str | None = None
            action = RepairAction.CLAIM_DELETION
            resolved_ids: list[str] = []
            remap_attempted = False
            second_status: EntailmentStatus | None = None

            if result.label == VerificationLabel.PARTIAL and controls.repair_mode == RepairMode.SELECTIVE:
                narrowed = cls._supported_narrowing(claim, evidence)
                if narrowed is not None:
                    replacement, second_result = narrowed
                    action = RepairAction.CLAIM_NARROWING
                    resolved_ids = second_result.cited_evidence_ids
                    second_status = EntailmentStatus.SUPPORTED
                else:
                    replacement = claim.text
                    action = RepairAction.NONE
                    resolved_ids = original_evidence_ids
                    second_status = EntailmentStatus.PARTIAL
            elif result.label == VerificationLabel.UNSUPPORTED and controls.repair_mode in {
                RepairMode.CITATION_REMAP_ONLY,
                RepairMode.SELECTIVE,
            }:
                used_record_ids = {id(item) for item in evidence}
                best: tuple[float, dict[str, Any], ClaimVerificationResult] | None = None
                remap_attempted = True
                for candidate in records:
                    if id(candidate) in used_record_ids:
                        continue
                    candidate_text = cls._evidence_text(candidate)
                    candidate_result = cls.verify_claim(
                        claim.claim_id,
                        claim.text,
                        [candidate_text],
                        cited_evidence_ids=[str(candidate["_evidence_id"])],
                    )
                    if candidate_result.label in {VerificationLabel.SUPPORTED, VerificationLabel.PARTIAL}:
                        score = candidate_result.confidence
                        if best is None or score > best[0]:
                            best = (score, candidate, candidate_result)
                if best is not None:
                    _, candidate, _ = best
                    remapped = cls._replace_citations(claim.text, int(candidate["ref_id"]))
                    second_result = cls.verify_claim(
                        claim.claim_id,
                        remapped,
                        [str(candidate["_evidence_text"])],
                        cited_evidence_ids=[str(candidate["_evidence_id"])],
                    )
                    if second_result.label in {VerificationLabel.SUPPORTED, VerificationLabel.PARTIAL} and remapped != claim.text:
                        replacement = remapped
                        action = RepairAction.CITATION_REMAP
                        resolved_ids = second_result.cited_evidence_ids
                        second_status = EntailmentStatus.SUPPORTED
                        candidate["_remapped"] = True

            if replacement is None:
                if controls.repair_mode != RepairMode.SELECTIVE:
                    continue
                replacement = ""
                action = RepairAction.CLAIM_DELETION
            if replacement == claim.text:
                continue
            replacements.append((claim.start, claim.end, replacement))
            edits.append(ClaimRepairRecord(
                claim_id=claim.claim_id,
                action=action,
                original_start=claim.start,
                original_end=claim.end,
                original_text=claim.text,
                replacement_text=replacement,
                initial_status=cls._status(result.label),
                second_pass_status=second_status,
                original_evidence_ids=original_evidence_ids,
                resolved_evidence_ids=resolved_ids,
                remap_attempted=remap_attempted,
            ))

        final_answer = answer
        for start, end, replacement in sorted(replacements, key=lambda edit: edit[0], reverse=True):
            final_answer = final_answer[:start] + replacement + final_answer[end:]

        if final_answer != answer:
            final_answer = re.sub(r"^\s*[-*+]\s*$\n?", "", final_answer, flags=re.MULTILINE)
            final_answer = re.sub(r"\n{3,}", "\n\n", final_answer).strip()

        final_claims, final_results, _ = cls._evaluate_answer(final_answer, by_ref, by_id, second_pass=True)
        retained_factual = [claim for claim in final_claims if claim.claim_type == "factual"]
        has_supported = any(
            claim.entailment_status in {EntailmentStatus.SUPPORTED, EntailmentStatus.PARTIAL}
            for claim in retained_factual
        )
        had_factual = any(claim.claim_type == "factual" for claim in initial_claims)
        abstained = had_factual and not has_supported

        if abstained and controls.abstain_on_no_supported_claims:
            final_answer = _ABSTENTION_TEXT
            edits = [ClaimRepairRecord(
                claim_id="ABSTAIN",
                action=RepairAction.ABSTAIN,
                original_start=0,
                original_end=len(answer),
                original_text=answer,
                replacement_text=final_answer,
                initial_status=(
                    initial_claims[0].entailment_status
                    if initial_claims else EntailmentStatus.UNSUPPORTED
                ),
                remap_attempted=any(edit.remap_attempted for edit in edits),
            )]
            final_claims, final_results, _ = cls._evaluate_answer(final_answer, by_ref, by_id, second_pass=True)
        else:
            abstained = False

        if any(edit.original_text == edit.replacement_text for edit in edits):
            raise AssertionError("Every recorded repair must change the addressed text")
        if bool(edits) != (final_answer != answer):
            raise AssertionError("Repair records and final text mutation disagree")

        used_numeric_refs = {
            int(ref)
            for claim in cls.parse_answer_into_claims(final_answer)
            for marker in claim.citations
            for ref in marker.reference_ids
            if ref.isdigit()
        }
        final_citations: list[dict[str, Any]] = []
        labels_by_ref: dict[int, list[ClaimVerificationResult]] = {}
        for parsed, result in zip(cls.parse_answer_into_claims(final_answer), final_results):
            for marker in parsed.citations:
                for ref in marker.reference_ids:
                    if ref.isdigit():
                        labels_by_ref.setdefault(int(ref), []).append(result)
        for ref_id in sorted(used_numeric_refs):
            record = by_ref.get(ref_id)
            if record is None:
                continue
            public = {key: value for key, value in record.items() if not key.startswith("_")}
            evaluations = labels_by_ref.get(ref_id, [])
            public["verification"] = (
                VerificationLabel.SUPPORTED.value
                if evaluations and all(item.label == VerificationLabel.SUPPORTED for item in evaluations)
                else VerificationLabel.UNSUPPORTED.value
            )
            public["confidence"] = min((item.confidence for item in evaluations), default=0.0)
            if record.get("_remapped"):
                public["repair_origin"] = "REMAPPED"
            final_citations.append(public)

        final_report = cls._report(
            final_answer,
            final_claims,
            edits=edits,
            second_pass=True,
            abstained=abstained,
        )
        final_report.final_verified_response = final_answer
        return VerificationOutcome(
            original_answer=answer,
            final_answer=final_answer,
            citations=final_citations,
            initial_results=initial_results,
            final_results=final_results,
            initial_report=initial_report,
            final_report=final_report,
            edits=edits,
            reverified=has_supported,
        )

    @classmethod
    def align_and_prune_citations(
        cls,
        answer: str,
        citations: list[dict[str, Any]],
        candidate_pool: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Compatibility wrapper for callers that only need aligned text and citations."""
        outcome = cls.verify_and_repair_detailed(answer, citations, candidate_pool, apply_repair=True)
        return outcome.final_answer, outcome.citations

    @classmethod
    def verify_and_repair_answer(
        cls,
        answer: str,
        citations: list[dict[str, Any]],
        candidate_pool: list[dict[str, Any]] | None = None,
        apply_repair: bool = True,
    ) -> tuple[str, list[dict[str, Any]], list[ClaimVerificationResult]]:
        """Compatibility tuple API backed by the detailed span-preserving outcome."""
        outcome = cls.verify_and_repair_detailed(answer, citations, candidate_pool, apply_repair)
        return outcome.final_answer, outcome.citations, outcome.final_results

    @classmethod
    def is_negative_or_disclaimer(cls, text: str) -> bool:
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
            r"\bdoes\s+not\s+provide\s+sufficient\s+evidence\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    @classmethod
    def compute_citation_metrics(
        cls,
        verified_claims: list[ClaimVerificationResult],
        gold_citations: list[dict[str, Any]] | list[str] | None = None,
    ) -> dict[str, float]:
        """Compute citation precision/recall/F1 and unsupported-claim rate."""
        if not verified_claims:
            return {
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "citation_f1": 1.0,
                "unsupported_claim_rate": 0.0,
                "total_claims": 0,
            }
        total_claims = len(verified_claims)
        supported_count = sum(claim.label == VerificationLabel.SUPPORTED for claim in verified_claims)
        unsupported_count = sum(
            claim.label in (VerificationLabel.PARTIAL, VerificationLabel.UNSUPPORTED, VerificationLabel.CONTRADICTED)
            for claim in verified_claims
        )
        precision = supported_count / total_claims
        unsupported_rate = unsupported_count / total_claims
        if gold_citations:
            gold_keys = {
                str(item.get("source_id") or item.get("page") or item) if isinstance(item, dict) else str(item)
                for item in gold_citations
            }
            cited_keys = {key for claim in verified_claims for key in claim.cited_evidence_ids}
            recall = len(gold_keys.intersection(cited_keys)) / max(1, len(gold_keys))
        else:
            recall = 1.0 if supported_count else 0.0
        citation_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "citation_precision": round(float(precision), 4),
            "citation_recall": round(float(recall), 4),
            "citation_f1": round(float(citation_f1), 4),
            "unsupported_claim_rate": round(float(unsupported_rate), 4),
            "total_claims": total_claims,
            "supported_claims": supported_count,
            "unsupported_claims": unsupported_count,
        }
