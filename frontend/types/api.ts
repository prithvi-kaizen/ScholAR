import type {
  Citation,
  NumericExecutionResult,
  ReasoningPathStep,
  VerificationReport,
} from "./paper";

export type NetworkMode = "acquisition-enabled" | "strict-local";

export type NetworkPolicyStatus = {
  mode: NetworkMode;
  external_network_allowed: boolean;
  local_model_endpoints_only: boolean;
  assets: Array<{ asset: string; available: boolean; detail: string }>;
  missing_assets: string[];
  actions: Array<{
    action: string;
    requires_external_network: boolean;
    allowed: boolean;
  }>;
};

export type ChatHistoryItem = {
  role: "user" | "assistant";
  content: string;
};

export type ChatRequest = {
  message: string;
  history: ChatHistoryItem[];
  secondary_paper_ids: string[];
  model?: string;
  capability_mode?: string;
  snippet_id?: string;
  snippet_page?: number;
  snippet_bbox?: [number, number, number, number];
  snippet_text?: string;
};

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  error?: boolean;
  message?: string;
  model?: string;
  route_type?: string;
  capability_mode?: string;
  abstained?: boolean;
  uncertainty_reason?: string;
  vision?: boolean;
  vision_fallback?: boolean;
  is_snippet?: boolean;
  snippet_id?: string | null;
  figure_id?: string | null;
  figure_label?: string | null;
  figure_image_url?: string | null;
  reasoning_level?: string;
  reasoning_steps?: ReasoningPathStep[];
  numeric_plan?: NumericExecutionResult | null;
  verification_report?: VerificationReport | null;
};

export type FigureMeta = {
  figure_id: string;
  figure_type: string;
  label: string;
  caption: string;
  page: number;
};

export type FigureListResponse = {
  paper_id: string;
  figures: FigureMeta[];
  count: number;
};

export type ExportReasoningResponse = {
  format: string;
  filename: string;
  content: string;
};

export type EvidenceNode = {
  node_id: string;
  document_id: string;
  page: number;
  section: string;
  modality: string;
  text_preview: string;
  reasoning_role: string;
  score?: number;
};

export type EvidenceEdge = {
  source_id: string;
  target_id: string;
  relation: string;
  description: string;
  weight?: number;
};

export type EvidenceGraph = {
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
  graph_id?: string;
  query?: string;
  reasoning_level?: string;
};

export type CrossDocumentReasoningResponse = {
  graph: EvidenceGraph;
  path: {
    query: string;
    reasoning_level: string;
    steps: ReasoningPathStep[];
    synthesized_rationale?: string;
  };
  retrieved_count: number;
};

export type SystemDiagnostic = {
  status: string;
  acceleration: {
    device: string;
    is_gpu_accelerated: boolean;
    torch_version: string;
  };
  memory: {
    total_ram_gb: number;
    available_ram_gb: number;
    hardware_tier: string;
    token_budget: number;
    max_evidence_blocks: number;
  };
  local_llm: {
    ollama_url: string;
    active_model: string;
    is_connected: boolean;
    mode: string;
  };
  storage: {
    ingested_papers_count: number;
    cached_embeddings_count: number;
    telemetry_traces_recorded: number;
  };
};

export type SubQuery = {
  subquery_id: string;
  query_text: string;
  target_sections?: string[];
  target_modality?: string;
  priority?: number;
  sufficiency_score?: number;
  is_grounded?: boolean;
  retrieved_evidence_ids?: string[];
};

export type TelemetryTrace = {
  trace_id: string;
  paper_id: string;
  query: string;
  reasoning_level: string;
  target_modalities: string[];
  subqueries: SubQuery[];
  reasoning_path: ReasoningPathStep[];
  numeric_plan: NumericExecutionResult | null;
  verification_report: VerificationReport | null;
  latency_ms: number;
  hardware_tier: string;
  timestamp: number;
};

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isNetworkPolicyStatus(value: unknown): value is NetworkPolicyStatus {
  return (
    isRecord(value) &&
    (value.mode === "acquisition-enabled" || value.mode === "strict-local") &&
    typeof value.external_network_allowed === "boolean" &&
    typeof value.local_model_endpoints_only === "boolean" &&
    isStringArray(value.missing_assets) &&
    Array.isArray(value.assets) &&
    value.assets.every(
      (asset) => isRecord(asset) && typeof asset.asset === "string" &&
        typeof asset.available === "boolean" && typeof asset.detail === "string"
    ) &&
    Array.isArray(value.actions) &&
    value.actions.every(
      (action) => isRecord(action) && typeof action.action === "string" &&
        typeof action.requires_external_network === "boolean" && typeof action.allowed === "boolean"
    )
  );
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isOptionalString(value: unknown): value is string | null | undefined {
  return value === undefined || value === null || typeof value === "string";
}

function isCitation(value: unknown): value is Citation {
  return (
    isRecord(value) &&
    typeof value.page === "number" &&
    typeof value.chunk_id === "string" &&
    typeof value.quote === "string"
  );
}

export function isReasoningPathStep(value: unknown): value is ReasoningPathStep {
  return (
    isRecord(value) &&
    typeof value.step_index === "number" &&
    typeof value.evidence_id === "string" &&
    typeof value.page === "number" &&
    typeof value.modality === "string" &&
    typeof value.role === "string" &&
    typeof value.claim_contribution === "string" &&
    (value.section === undefined || typeof value.section === "string") &&
    (value.reasoning_mode === undefined || typeof value.reasoning_mode === "string") &&
    (value.subgoal === undefined || typeof value.subgoal === "string")
  );
}

function isNumericExecutionResult(value: unknown): value is NumericExecutionResult {
  return (
    isRecord(value) &&
    typeof value.operation === "string" &&
    typeof value.computed_value === "number" &&
    typeof value.formatted_value === "string" &&
    typeof value.formatted_statement === "string" &&
    typeof value.is_exact === "boolean"
  );
}

const verificationLabels = new Set([
  "SUPPORTED",
  "PARTIAL",
  "PARTIALLY_SUPPORTED",
  "UNSUPPORTED",
  "CONTRADICTED",
]);
const repairActions = new Set([
  "none",
  "citation_remap",
  "claim_narrowing",
  "numeric_correction",
  "claim_deletion",
  "abstain",
]);

function isVerificationLabel(value: unknown): boolean {
  return typeof value === "string" && verificationLabels.has(value);
}

function isRepairAction(value: unknown): boolean {
  return typeof value === "string" && repairActions.has(value);
}

function isOptionalOffset(value: unknown): boolean {
  return value === undefined || value === null || (Number.isInteger(value) && (value as number) >= 0);
}

function isCitationSpan(value: unknown): boolean {
  return (
    isRecord(value) &&
    Number.isInteger(value.start) &&
    Number.isInteger(value.end) &&
    (value.start as number) >= 0 &&
    (value.end as number) >= (value.start as number) &&
    typeof value.marker === "string" &&
    isStringArray(value.reference_ids) &&
    isStringArray(value.evidence_ids)
  );
}

function isEvidenceProvenance(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.evidence_id === "string" &&
    (value.ref_id === undefined || value.ref_id === null || Number.isInteger(value.ref_id)) &&
    isOptionalString(value.source_paper_id) &&
    isOptionalString(value.document_id) &&
    (value.page === undefined || value.page === null || Number.isInteger(value.page))
  );
}

function isClaimRepairRecord(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.claim_id === "string" &&
    isRepairAction(value.action) &&
    Number.isInteger(value.original_start) &&
    Number.isInteger(value.original_end) &&
    (value.original_start as number) >= 0 &&
    (value.original_end as number) >= (value.original_start as number) &&
    typeof value.original_text === "string" &&
    typeof value.replacement_text === "string" &&
    value.original_text !== value.replacement_text &&
    isVerificationLabel(value.initial_status) &&
    (value.second_pass_status === undefined || value.second_pass_status === null || isVerificationLabel(value.second_pass_status)) &&
    isStringArray(value.original_evidence_ids) &&
    isStringArray(value.resolved_evidence_ids) &&
    typeof value.remap_attempted === "boolean"
  );
}

function isAtomicClaim(value: unknown): boolean {
  if (!isRecord(value)) return false;
  if (
    typeof value.claim_id !== "string" ||
    typeof value.text !== "string" ||
    !isStringArray(value.cited_evidence_ids) ||
    !isVerificationLabel(value.entailment_status) ||
    typeof value.confidence_score !== "number" ||
    typeof value.rationale !== "string" ||
    !isRepairAction(value.repair_action) ||
    !isOptionalOffset(value.start) ||
    !isOptionalOffset(value.end) ||
    (typeof value.start === "number" && typeof value.end === "number" && value.end < value.start)
  ) return false;
  return (
    (value.repaired_text === undefined || value.repaired_text === null || typeof value.repaired_text === "string") &&
    (value.citation_spans === undefined || (Array.isArray(value.citation_spans) && value.citation_spans.every(isCitationSpan))) &&
    (value.normalized_text === undefined || typeof value.normalized_text === "string") &&
    (value.claim_type === undefined || typeof value.claim_type === "string") &&
    (value.resolved_evidence === undefined ||
      (Array.isArray(value.resolved_evidence) && value.resolved_evidence.every(isEvidenceProvenance))) &&
    (value.first_pass_status === undefined || value.first_pass_status === null || isVerificationLabel(value.first_pass_status)) &&
    (value.second_pass_status === undefined || value.second_pass_status === null || isVerificationLabel(value.second_pass_status)) &&
    isOptionalOffset(value.final_start) &&
    isOptionalOffset(value.final_end)
  );
}

function isVerificationReport(value: unknown): value is VerificationReport {
  return (
    isRecord(value) &&
    typeof value.overall_supported === "boolean" &&
    typeof value.supported_count === "number" &&
    (value.partial_count === undefined || typeof value.partial_count === "number") &&
    typeof value.unsupported_count === "number" &&
    typeof value.contradicted_count === "number" &&
    typeof value.has_abstained === "boolean" &&
    isOptionalString(value.abstention_reason) &&
    isOptionalString(value.final_verified_response) &&
    (value.second_pass_completed === undefined || typeof value.second_pass_completed === "boolean") &&
    (value.claims === undefined || (Array.isArray(value.claims) && value.claims.every(isAtomicClaim))) &&
    (value.edits === undefined || (Array.isArray(value.edits) && value.edits.every(isClaimRepairRecord)))
  );
}

export function isChatResponse(value: unknown): value is ChatResponse {
  return (
    isRecord(value) &&
    typeof value.answer === "string" &&
    Array.isArray(value.citations) &&
    value.citations.every(isCitation) &&
    (value.error === undefined || typeof value.error === "boolean") &&
    isOptionalString(value.message) &&
    isOptionalString(value.route_type) &&
    isOptionalString(value.capability_mode) &&
    isOptionalString(value.uncertainty_reason) &&
    (value.reasoning_steps === undefined ||
      (Array.isArray(value.reasoning_steps) && value.reasoning_steps.every(isReasoningPathStep))) &&
    (value.numeric_plan === undefined || value.numeric_plan === null || isNumericExecutionResult(value.numeric_plan)) &&
    (value.verification_report === undefined ||
      value.verification_report === null ||
      isVerificationReport(value.verification_report))
  );
}

export function getApiErrorMessage(value: unknown, fallback: string): string {
  if (!isRecord(value)) return fallback;
  if (typeof value.detail === "string") return value.detail;
  if (isRecord(value.detail) && typeof value.detail.message === "string") return value.detail.message;
  if (typeof value.message === "string") return value.message;
  return fallback;
}

function isFigureMeta(value: unknown): value is FigureMeta {
  return (
    isRecord(value) &&
    typeof value.figure_id === "string" &&
    typeof value.figure_type === "string" &&
    typeof value.label === "string" &&
    typeof value.caption === "string" &&
    typeof value.page === "number"
  );
}

export function isFigureListResponse(value: unknown): value is FigureListResponse {
  return (
    isRecord(value) &&
    typeof value.paper_id === "string" &&
    Array.isArray(value.figures) &&
    value.figures.every(isFigureMeta) &&
    typeof value.count === "number"
  );
}

export function isExportReasoningResponse(value: unknown): value is ExportReasoningResponse {
  return (
    isRecord(value) &&
    typeof value.format === "string" &&
    typeof value.filename === "string" &&
    typeof value.content === "string"
  );
}

function isEvidenceNode(value: unknown): value is EvidenceNode {
  return (
    isRecord(value) &&
    typeof value.node_id === "string" &&
    typeof value.document_id === "string" &&
    typeof value.page === "number" &&
    typeof value.section === "string" &&
    typeof value.modality === "string" &&
    typeof value.text_preview === "string" &&
    typeof value.reasoning_role === "string"
  );
}

function isEvidenceEdge(value: unknown): value is EvidenceEdge {
  return (
    isRecord(value) &&
    typeof value.source_id === "string" &&
    typeof value.target_id === "string" &&
    typeof value.relation === "string" &&
    typeof value.description === "string"
  );
}

export function isCrossDocumentReasoningResponse(value: unknown): value is CrossDocumentReasoningResponse {
  if (!isRecord(value) || !isRecord(value.graph) || !isRecord(value.path)) return false;
  return (
    Array.isArray(value.graph.nodes) &&
    value.graph.nodes.every(isEvidenceNode) &&
    Array.isArray(value.graph.edges) &&
    value.graph.edges.every(isEvidenceEdge) &&
    typeof value.path.query === "string" &&
    typeof value.path.reasoning_level === "string" &&
    Array.isArray(value.path.steps) &&
    value.path.steps.every(isReasoningPathStep) &&
    typeof value.retrieved_count === "number"
  );
}

export function isSystemDiagnostic(value: unknown): value is SystemDiagnostic {
  if (
    !isRecord(value) ||
    !isRecord(value.acceleration) ||
    !isRecord(value.memory) ||
    !isRecord(value.local_llm) ||
    !isRecord(value.storage)
  ) {
    return false;
  }
  return (
    typeof value.status === "string" &&
    typeof value.acceleration.device === "string" &&
    typeof value.acceleration.is_gpu_accelerated === "boolean" &&
    typeof value.acceleration.torch_version === "string" &&
    typeof value.memory.total_ram_gb === "number" &&
    typeof value.memory.available_ram_gb === "number" &&
    typeof value.memory.hardware_tier === "string" &&
    typeof value.memory.token_budget === "number" &&
    typeof value.memory.max_evidence_blocks === "number" &&
    typeof value.local_llm.ollama_url === "string" &&
    typeof value.local_llm.active_model === "string" &&
    typeof value.local_llm.is_connected === "boolean" &&
    typeof value.local_llm.mode === "string" &&
    typeof value.storage.ingested_papers_count === "number" &&
    typeof value.storage.cached_embeddings_count === "number" &&
    typeof value.storage.telemetry_traces_recorded === "number"
  );
}

function isSubQuery(value: unknown): value is SubQuery {
  return (
    isRecord(value) &&
    typeof value.subquery_id === "string" &&
    typeof value.query_text === "string"
  );
}

export function isTelemetryTrace(value: unknown): value is TelemetryTrace {
  return (
    isRecord(value) &&
    typeof value.trace_id === "string" &&
    typeof value.paper_id === "string" &&
    typeof value.query === "string" &&
    typeof value.reasoning_level === "string" &&
    isStringArray(value.target_modalities) &&
    Array.isArray(value.subqueries) &&
    value.subqueries.every(isSubQuery) &&
    Array.isArray(value.reasoning_path) &&
    value.reasoning_path.every(isReasoningPathStep) &&
    (value.numeric_plan === null || isNumericExecutionResult(value.numeric_plan)) &&
    (value.verification_report === null || isVerificationReport(value.verification_report)) &&
    typeof value.latency_ms === "number" &&
    typeof value.hardware_tier === "string" &&
    typeof value.timestamp === "number"
  );
}
