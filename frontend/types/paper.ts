export type Paper = {
  id: string;
  local_id?: string;
  title: string;
  authors: string[];
  year: string;
  published?: string;
  summary: string;
  categories: string[];
  pdf_url: string;
  abs_url: string;
  source?: "arxiv" | "upload";
  filename?: string;
};

export type StudyGoal = {
  id: string;
  title: string;
  description: string;
  phase?: "Foundation" | "Architecture" | "Benchmarks" | "Implementation";
  difficulty?: "Foundational" | "Intermediate" | "Advanced";
  estimated_minutes?: number;
  target_evidence?: string[];
  key_takeaways?: string[];
  source_pages: number[];
  subquestions?: {
    id: string;
    question: string;
    evidence_chunks?: {
      chunk_id: string;
      page: number;
      section_title?: string;
      chunk_type?: string;
      quote?: string;
    }[];
  }[];
  status: "not_started" | "in_progress" | "done";
};

export type VerificationLabel =
  | "SUPPORTED"
  | "PARTIAL"
  | "PARTIALLY_SUPPORTED" // accepted only for legacy persisted traces
  | "UNSUPPORTED"
  | "CONTRADICTED";

export type VisualSubregion = {
  region_id?: string;
  role?: string;
  bbox: { x0: number; y0: number; x1: number; y1: number };
  verification?: VerificationLabel;
};

export type Citation = {
  ref_id?: number;
  page: number;
  chunk_id: string;
  section_title?: string;
  chunk_type?: string;
  quote: string;
  /** Source-scoped provenance for multi-document evidence. */
  source_paper_id?: string;
  document_id?: string;
  source_evidence_id?: string;
  /** Figure/table grounding fields. */
  is_figure?: boolean;
  figure_id?: string;
  image_file?: string;
  image_relpath?: string;
  image_url?: string;
  is_page_visual?: boolean;
  label?: string;
  caption?: string;
  verification?: VerificationLabel;
  confidence?: number;
  bbox_normalized?: { x0: number; y0: number; x1: number; y1: number };
  subregions?: VisualSubregion[];
};

export type ReasoningPathStep = {
  step_index: number;
  evidence_id: string;
  section?: string;
  page: number;
  modality: string;
  role: string;
  reasoning_mode?: string;
  subgoal?: string;
  claim_contribution: string;
  source_paper_id?: string;
  document_id?: string;
  source_evidence_id?: string;
};

export type NumericExecutionResult = {
  operation: string;
  computed_value: number;
  formatted_value: string;
  formatted_statement: string;
  is_exact: boolean;
  evidence_ids?: string[];
};

export type CitationSpan = {
  start: number;
  end: number;
  marker: string;
  reference_ids: string[];
  evidence_ids: string[];
};

export type EvidenceProvenance = {
  evidence_id: string;
  ref_id?: number | null;
  source_paper_id?: string | null;
  document_id?: string | null;
  page?: number | null;
  region?: Record<string, unknown> | number[] | null;
};

export type RepairAction =
  | "none"
  | "citation_remap"
  | "claim_narrowing"
  | "numeric_correction"
  | "claim_deletion"
  | "abstain";

export type ClaimRepairRecord = {
  claim_id: string;
  action: RepairAction;
  original_start: number;
  original_end: number;
  original_text: string;
  replacement_text: string;
  initial_status: Exclude<VerificationLabel, "PARTIALLY_SUPPORTED">;
  second_pass_status?: Exclude<VerificationLabel, "PARTIALLY_SUPPORTED"> | null;
  original_evidence_ids: string[];
  resolved_evidence_ids: string[];
  remap_attempted: boolean;
};

export type AtomicClaim = {
  claim_id: string;
  text: string;
  cited_evidence_ids: string[];
  entailment_status: VerificationLabel;
  confidence_score: number;
  rationale: string;
  repaired_text?: string | null;
  repair_action: RepairAction;
  start?: number | null;
  end?: number | null;
  citation_spans?: CitationSpan[];
  normalized_text?: string;
  claim_type?: string;
  resolved_evidence?: EvidenceProvenance[];
  first_pass_status?: VerificationLabel | null;
  second_pass_status?: VerificationLabel | null;
  final_start?: number | null;
  final_end?: number | null;
};

export type VerificationReport = {
  claims?: AtomicClaim[];
  overall_supported: boolean;
  supported_count: number;
  partial_count?: number;
  unsupported_count: number;
  contradicted_count: number;
  has_abstained: boolean;
  abstention_reason?: string | null;
  final_verified_response?: string;
  edits?: ClaimRepairRecord[];
  second_pass_completed?: boolean;
  scorer?: {
    backend: string;
    version: string;
    thresholds_calibrated: boolean;
    supported_threshold: number;
    partial_threshold: number;
  };
};

export type ModelCapability = {
  model_id: string;
  display_name: string;
  backend: string;
  supports_vision: boolean;
  supports_text: boolean;
  context_length: number;
  capability_mode: string;
};

export interface CustomSnippet {
  id: string;
  page: number;
  bbox: [number, number, number, number]; // [x0, y0, x1, y1] normalized
  imageUrl: string;
  text?: string;
}

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  error?: boolean;
  abstained?: boolean;
  uncertainty_reason?: string;
  route_type?: string;
  capability_mode?: string;
  /** Vision grounding fields: set when the answer was generated from a figure or snippet. */
  vision?: boolean;
  vision_fallback?: boolean;
  is_snippet?: boolean;
  snippet_id?: string;
  figure_id?: string;
  figure_label?: string;
  figure_image_url?: string;
  /** Which model produced the response, e.g. "qwen3.5:9b". */
  model?: string;
  reasoning_level?: string;
  reasoning_steps?: ReasoningPathStep[];
  numeric_plan?: NumericExecutionResult;
  verification_report?: VerificationReport;
};
