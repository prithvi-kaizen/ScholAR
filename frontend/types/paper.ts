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

export type VisualSubregion = {
  region_id?: string;
  role?: string;
  bbox: { x0: number; y0: number; x1: number; y1: number };
  verification?: "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED";
};

export type Citation = {
  ref_id?: number;
  page: number;
  chunk_id: string;
  section_title?: string;
  chunk_type?: string;
  quote: string;
  /** For multi-document mode: the local_id of the paper this chunk came from */
  source_paper_id?: string;
  /** Figure/table grounding fields */
  is_figure?: boolean;
  figure_id?: string;
  image_file?: string;
  label?: string;
  caption?: string;
  verification?: "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED";
  confidence?: number;
  bbox_normalized?: { x0: number; y0: number; x1: number; y1: number };
  subregions?: VisualSubregion[];
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
  /** Vision grounding fields: set when the answer was generated from a figure or snippet */
  vision?: boolean;
  vision_fallback?: boolean;
  is_snippet?: boolean;
  snippet_id?: string;
  figure_id?: string;
  figure_label?: string;
  figure_image_url?: string;
  /** Which model produced the response, e.g. "qwen3.5:9b" */
  model?: string;
  /** Multi-Level Reasoning fields */
  reasoning_level?: string;
  reasoning_steps?: {
    step_index: number;
    evidence_id: string;
    section?: string;
    page: number;
    modality: string;
    role: string;
    claim_contribution: string;
  }[];
  numeric_plan?: {
    operation: string;
    computed_value: number;
    formatted_value: string;
    formatted_statement: string;
    is_exact: boolean;
  };
  verification_report?: {
    overall_supported: boolean;
    supported_count: number;
    unsupported_count: number;
    contradicted_count: number;
    has_abstained: boolean;
    abstention_reason?: string;
  };
};

