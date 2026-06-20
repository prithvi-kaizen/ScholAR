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

export type Citation = {
  ref_id?: number;
  page: number;
  chunk_id: string;
  section_title?: string;
  chunk_type?: string;
  quote: string;
  /** For multi-document mode: the local_id of the paper this chunk came from */
  source_paper_id?: string;
};

export type WebResult = {
  id: string;
  title: string;
  url: string;
  snippet: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  web_results?: WebResult[];
  used_web_search?: boolean;
  provider_error?: "groq_rate_limit" | string;
  retry_text?: string;
};
