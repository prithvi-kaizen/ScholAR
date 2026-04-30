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
};

export type StudyGoal = {
  id: string;
  title: string;
  description: string;
  source_pages: number[];
  status: "not_started" | "in_progress" | "done";
};

export type Citation = {
  page: number;
  chunk_id: string;
  quote: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};
