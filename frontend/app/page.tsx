"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ActionTiles } from "../components/ActionTiles";
import { Navbar } from "../components/Navbar";
import { PaperCard } from "../components/PaperCard";
import { PaperModal } from "../components/PaperModal";
import { SearchBar } from "../components/SearchBar";
import { useNetworkPolicy } from "../lib/networkPolicy";
import { getApiErrorMessage } from "../types/api";
import type { Paper } from "../types/paper";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const RECENT_KEY = "scholar_recent";
const BOOKMARK_KEY = "scholar_bookmarks";

const editorPicks: Paper[] = [
  {
    id: "1706.03762",
    title: "Attention Is All You Need",
    authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    year: "2017",
    summary:
      "The Transformer architecture replaces recurrent sequence modeling with attention mechanisms and becomes a foundation for modern language models.",
    categories: ["cs.CL", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/1706.03762",
    abs_url: "https://arxiv.org/abs/1706.03762"
  },
  {
    id: "1810.04805",
    title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
    year: "2018",
    summary:
      "BERT pre-trains deep bidirectional representations from unlabeled text via masked language modeling, setting new state-of-the-art results across NLP tasks.",
    categories: ["cs.CL"],
    pdf_url: "https://arxiv.org/pdf/1810.04805",
    abs_url: "https://arxiv.org/abs/1810.04805"
  },
  {
    id: "1512.03385",
    title: "Deep Residual Learning for Image Recognition",
    authors: ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
    year: "2015",
    summary:
      "ResNet introduces residual connections that make it possible to train dramatically deeper convolutional networks, becoming a backbone for modern vision models.",
    categories: ["cs.CV"],
    pdf_url: "https://arxiv.org/pdf/1512.03385",
    abs_url: "https://arxiv.org/abs/1512.03385"
  },
  {
    id: "2005.14165",
    title: "Language Models are Few-Shot Learners",
    authors: ["Tom B. Brown", "Benjamin Mann", "Nick Ryder"],
    year: "2020",
    summary:
      "GPT-3 shows that scaling autoregressive language models to 175B parameters enables strong few-shot performance across tasks without gradient updates.",
    categories: ["cs.CL"],
    pdf_url: "https://arxiv.org/pdf/2005.14165",
    abs_url: "https://arxiv.org/abs/2005.14165"
  },
  {
    id: "2005.11401",
    title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    authors: ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
    year: "2020",
    summary:
      "RAG combines parametric generation with non-parametric retrieval for tasks where factual grounding matters.",
    categories: ["cs.CL", "cs.AI"],
    pdf_url: "https://arxiv.org/pdf/2005.11401",
    abs_url: "https://arxiv.org/abs/2005.11401"
  },
  {
    id: "2006.11239",
    title: "Denoising Diffusion Probabilistic Models",
    authors: ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"],
    year: "2020",
    summary:
      "DDPMs generate high-quality images by learning to reverse a gradual noising process, laying the groundwork for modern diffusion-based generative models.",
    categories: ["cs.LG", "cs.CV"],
    pdf_url: "https://arxiv.org/pdf/2006.11239",
    abs_url: "https://arxiv.org/abs/2006.11239"
  },
  {
    id: "2010.11929",
    title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    authors: ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
    year: "2020",
    summary:
      "The Vision Transformer (ViT) applies a pure Transformer directly to sequences of image patches, matching or beating CNNs on image classification at scale.",
    categories: ["cs.CV", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/2010.11929",
    abs_url: "https://arxiv.org/abs/2010.11929"
  },
  {
    id: "2201.11903",
    title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    authors: ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans"],
    year: "2022",
    summary:
      "Prompting large language models with intermediate reasoning steps substantially improves performance on arithmetic, commonsense, and symbolic reasoning tasks.",
    categories: ["cs.CL", "cs.AI"],
    pdf_url: "https://arxiv.org/pdf/2201.11903",
    abs_url: "https://arxiv.org/abs/2201.11903"
  },
  {
    id: "2203.02155",
    title: "Training Language Models to Follow Instructions with Human Feedback",
    authors: ["Long Ouyang", "Jeff Wu", "Xu Jiang"],
    year: "2022",
    summary:
      "InstructGPT fine-tunes language models with human feedback (RLHF) to better follow user instructions, improving helpfulness and truthfulness over raw scaling.",
    categories: ["cs.CL", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/2203.02155",
    abs_url: "https://arxiv.org/abs/2203.02155"
  },
  {
    id: "2302.13971",
    title: "LLaMA: Open and Efficient Foundation Language Models",
    authors: ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard"],
    year: "2023",
    summary:
      "LLaMA introduces a family of efficient foundation language models trained on public data at multiple scales.",
    categories: ["cs.CL", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/2302.13971",
    abs_url: "https://arxiv.org/abs/2302.13971"
  }
];

const testingPapers: Paper[] = [
  {
    id: "1406.2661",
    title: "Generative Adversarial Nets (GAN)",
    authors: ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza", "Yoshua Bengio"],
    year: "2014",
    summary:
      "A new framework for estimating generative models via an adversarial minimax game between generator G and discriminator D.",
    categories: ["cs.LG", "stat.ML"],
    pdf_url: "https://arxiv.org/pdf/1406.2661",
    abs_url: "https://arxiv.org/abs/1406.2661"
  },
  {
    id: "1412.6980",
    title: "Adam: A Method for Stochastic Optimization",
    authors: ["Diederik P. Kingma", "Jimmy Ba"],
    year: "2014",
    summary:
      "A method for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.",
    categories: ["cs.LG"],
    pdf_url: "https://arxiv.org/pdf/1412.6980",
    abs_url: "https://arxiv.org/abs/1412.6980"
  },
  {
    id: "2112.10752",
    title: "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)",
    authors: ["Robin Rombach", "Andreas Blattmann", "Dominik Lorenz", "Patrick Esser", "Björn Ommer"],
    year: "2021",
    summary:
      "By decomposing the image formation process into denoising autoencoders in latent space, diffusion models achieve state-of-the-art synthesis with lower compute.",
    categories: ["cs.CV", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/2112.10752",
    abs_url: "https://arxiv.org/abs/2112.10752"
  },
  {
    id: "1706.03762",
    title: "Attention Is All You Need (Transformer)",
    authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones"],
    year: "2017",
    summary:
      "The dominant sequence transduction models are replaced by the Transformer, relying entirely on self-attention mechanisms to draw global dependencies.",
    categories: ["cs.CL", "cs.LG"],
    pdf_url: "https://arxiv.org/pdf/1706.03762",
    abs_url: "https://arxiv.org/abs/1706.03762"
  },
  {
    id: "2406.08394",
    title: "VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model",
    authors: ["Jiannan Wu", "Zheng Ma", "Zhenyu Yang", "Lei Zhang"],
    year: "2024",
    summary:
      "An end-to-end generalist multimodal large language model unifying visual perception, understanding, and generation with interactive visual anchor tokens.",
    categories: ["cs.CV", "cs.AI"],
    pdf_url: "https://arxiv.org/pdf/2406.08394",
    abs_url: "https://arxiv.org/abs/2406.08394"
  },
  {
    id: "2104.08663",
    title: "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
    authors: ["Nandan Thakur", "Nils Reimers", "Andreas Rücklé", "Iryna Gurevych"],
    year: "2021",
    summary:
      "A heterogeneous benchmark containing 18 diverse datasets across 9 retrieval tasks to evaluate zero-shot generalization of search models.",
    categories: ["cs.IR", "cs.CL"],
    pdf_url: "https://arxiv.org/pdf/2104.08663",
    abs_url: "https://arxiv.org/abs/2104.08663"
  },
  {
    id: "2603.14257",
    title: "Automatic Inter-document Multi-hop Scientific QA Generation",
    authors: ["Seungmin Lee", "Dongha Kim", "Yuni Jeon", "Min Song"],
    year: "2026",
    summary:
      "An automated framework to generate multi-document, multi-hop scientific QA datasets leveraging LLMs and cross-document citation embedding alignment.",
    categories: ["cs.CL", "cs.AI"],
    pdf_url: "https://arxiv.org/pdf/2603.14257",
    abs_url: "https://arxiv.org/abs/2603.14257"
  },
  {
    id: "2025.emnlp-main.77",
    title: "MEBench: Benchmarking Large Language Models for Cross-Document Multi-Entity Question Answering",
    authors: ["Yao Zhang", "Sheng Shen", "Dan Roth", "Heng Ji"],
    year: "2025",
    summary:
      "A benchmark evaluating LLMs on cross-document multi-entity question answering across comparative, statistical, and relational reasoning tasks.",
    categories: ["cs.CL", "cs.AI"],
    pdf_url: "https://aclanthology.org/2025.emnlp-main.77.pdf",
    abs_url: "https://aclanthology.org/2025.emnlp-main.77"
  },
  {
    id: "yale_thesis_1003",
    title: "Towards Multi-Modal Multi-Document Understanding Capabilities in Foundation Models (M3SciQA)",
    authors: ["Chuhan Li", "Alexander R. Fabbri", "Arman Cohan"],
    year: "2024",
    summary:
      "Explores multi-modal scientific representation, cross-document reasoning, and the M3SciQA benchmark with SPACECUE visual prompting for foundation models.",
    categories: ["cs.CV", "cs.CL"],
    pdf_url: "https://arxiv.org/pdf/2406.03716",
    abs_url: "https://arxiv.org/abs/2406.03716"
  },
  {
    id: "2410.00526",
    title: "Conversational QA in Multi-instructional Documents",
    authors: ["Yixuan Su", "Kexin Wang", "Zuchao Li", "Linfeng Song"],
    year: "2024",
    summary:
      "Formulates and benchmarks conversational question answering across multi-instructional documents with dialogue state tracking.",
    categories: ["cs.CL", "cs.AI"],
    pdf_url: "https://arxiv.org/pdf/2410.00526",
    abs_url: "https://arxiv.org/abs/2410.00526"
  }
];

function readPapers(key: string): Paper[] {
  try {
    const stored = window.localStorage.getItem(key);
    return stored ? (JSON.parse(stored) as Paper[]) : [];
  } catch {
    return [];
  }
}

function writePapers(key: string, papers: Paper[]) {
  window.localStorage.setItem(key, JSON.stringify(papers));
}

export default function HomePage() {
  const router = useRouter();
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const [papers, setPapers] = useState<Paper[]>(editorPicks);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [recentlyViewed, setRecentlyViewed] = useState<Paper[]>([]);
  const [bookmarks, setBookmarks] = useState<Paper[]>([]);
  const [activeTab, setActiveTab] = useState("Editor's Picks");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStep, setUploadStep] = useState(0);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState("");
  const { policy } = useNetworkPolicy();

  const uploadSteps = [
    "Uploading PDF",
    "Extracting pages",
    "Building study chunks",
    "Creating study plan",
    "Opening workspace"
  ];

  useEffect(() => {
    setRecentlyViewed(readPapers(RECENT_KEY));
    setBookmarks(readPapers(BOOKMARK_KEY));
  }, []);

  const visiblePapers = useMemo(() => {
    if (searched) return papers;
    if (activeTab === "Testing") return testingPapers;
    if (activeTab === "Popular") return [...editorPicks].reverse();
    if (activeTab === "New") return [...editorPicks].sort((a, b) => Number(b.year) - Number(a.year));
    return editorPicks;
  }, [activeTab, papers, searched]);

  function rememberPaper(paper: Paper) {
    const next = [paper, ...recentlyViewed.filter((item) => item.id !== paper.id)].slice(0, 6);
    setRecentlyViewed(next);
    writePapers(RECENT_KEY, next);
  }

  function handleSelect(paper: Paper) {
    setSelectedPaper(paper);
    rememberPaper(paper);
  }

  function bookmarkPaper(paper: Paper) {
    const exists = bookmarks.some((item) => item.id === paper.id);
    const next = exists ? bookmarks.filter((item) => item.id !== paper.id) : [paper, ...bookmarks];
    setBookmarks(next);
    writePapers(BOOKMARK_KEY, next);
  }

  function openBookmarks() {
    document.getElementById("bookmarks")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function uploadPdf(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", file.name.replace(/\.pdf$/i, ""));

    setUploading(true);
    setUploadStep(0);
    setError("");
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), 90_000);
    try {
      setUploadStep(0);
      const response = await fetch(`${backendUrl}/api/papers/upload`, {
        method: "POST",
        body: formData,
        signal: timeoutController.signal
      });
      if (!response.ok) {
        const payload: unknown = await response.json().catch(() => ({}));
        throw new Error(getApiErrorMessage(payload, "Upload failed"));
      }
      setUploadStep(1);
      const payload = (await response.json()) as { paper_id: string; metadata: Paper };
      setUploadStep(2);
      rememberPaper(payload.metadata);
      setUploadStep(3);
      await fetch(`${backendUrl}/api/papers/${encodeURIComponent(payload.paper_id)}/study-goals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: true })
      }).catch(() => null);
      setUploadStep(4);
      router.push(`/paper/${encodeURIComponent(payload.paper_id)}`);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        setError("Upload timed out. Please try again.");
      } else {
        setError(caught instanceof Error ? caught.message : "Upload failed");
      }
    } finally {
      clearTimeout(timeoutId);
      setUploading(false);
    }
  }

  async function search(query: string) {
    const trimmed = query.trim();
    if (!trimmed) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${backendUrl}/api/search?q=${encodeURIComponent(trimmed)}&max_results=12`);
      if (!response.ok) {
        const payload: unknown = await response.json().catch(() => ({}));
        throw new Error(getApiErrorMessage(payload, "Search failed"));
      }
      const payload = (await response.json()) as { papers: Paper[] };
      setPapers(payload.papers ?? []);
      setSearched(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search failed");
      setSearched(true);
      setPapers([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-ink text-white">
      <Navbar />
      <section className="mx-auto w-full max-w-7xl px-5 py-8">
        <div className="mb-8">
          <SearchBar onSearch={search} loading={loading} />
          {policy ? (
            <p className="mt-2 text-xs text-zinc-400">
              {policy.mode === "strict-local"
                ? "Strict-local mode: search returns prepared local matches only; arXiv search and uncached PDF acquisition are blocked."
                : "Acquisition-enabled mode: search and uncached paper preparation may contact arXiv or the paper host."}
            </p>
          ) : null}
        </div>

        <div className="mb-8">
          <ActionTiles onUploadClick={() => uploadInputRef.current?.click()} onBookmarksClick={openBookmarks} />
          <input ref={uploadInputRef} type="file" accept="application/pdf" onChange={uploadPdf} className="hidden" />
        </div>

        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex rounded-lg border border-line bg-panel p-1">
            {["Editor's Picks", "Popular", "New", "Testing"].map((tab) => (
              <button
                key={tab}
                onClick={() => {
                  setActiveTab(tab);
                  setSearched(false);
                }}
                className={`rounded-md px-4 py-2 text-sm transition ${
                  activeTab === tab && !searched ? "bg-white text-black font-medium" : "text-zinc-400 hover:text-white"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
          {searched ? <span className="text-sm text-zinc-500">{papers.length} arXiv results</span> : null}
        </div>

        {uploading ? (
          <div className="fixed inset-0 z-50 grid place-items-center bg-black/80 px-4 backdrop-blur-sm">
            <div className="w-full max-w-lg rounded-lg border border-line bg-panel p-6 shadow-glow">
              <div className="mb-5 flex items-center gap-4">
                <div className="relative h-14 w-14 shrink-0">
                  <div className="absolute inset-0 rounded-full border-2 border-acid/20" />
                  <div className="absolute inset-0 animate-spin rounded-full border-2 border-acid border-t-transparent" />
                  <div className="absolute inset-3 rounded-full bg-acid/15" />
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-acid">Preparing study workspace</p>
                  <h2 className="mt-1 text-xl font-semibold text-white">{uploadSteps[uploadStep]}</h2>
                  <p className="mt-1 text-sm text-zinc-400">Extracting the paper and creating one canonical study plan.</p>
                </div>
              </div>
              <div className="mb-4 h-2 overflow-hidden rounded-full bg-white/10">
                <div className="h-full rounded-full bg-acid transition-all duration-500" style={{ width: `${((uploadStep + 1) / uploadSteps.length) * 100}%` }} />
              </div>
              <div className="space-y-2">
                {uploadSteps.map((step, index) => (
                  <div key={step} className={`flex items-center gap-2 text-sm ${index <= uploadStep ? "text-zinc-100" : "text-zinc-600"}`}>
                    <span className={`h-2 w-2 rounded-full ${index <= uploadStep ? "bg-acid" : "bg-zinc-700"}`} />
                    {step}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : null}
        {error ? <div className="mb-5 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div> : null}

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {[...Array(8)].map((_, index) => (
              <div key={index} className="h-48 animate-pulse rounded-lg border border-line bg-panel" />
            ))}
          </div>
        ) : visiblePapers.length ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {visiblePapers.map((paper) => (
              <PaperCard key={paper.id} paper={paper} onSelect={handleSelect} />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-line bg-panel px-6 py-12 text-center text-zinc-400">
            No papers found. Try a different arXiv query.
          </div>
        )}

        <section className="mt-10">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Recently Viewed Papers</h2>
            <span className="text-xs text-zinc-500">Stored locally</span>
          </div>
          {recentlyViewed.length ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {recentlyViewed.slice(0, 3).map((paper) => (
                <PaperCard key={`recent-${paper.id}`} paper={paper} onSelect={handleSelect} />
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-line bg-panel px-5 py-6 text-sm text-zinc-500">
              Open a paper to start your local reading trail.
            </div>
          )}
        </section>

        <section id="bookmarks" className="mt-10 scroll-mt-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Bookmarked Papers</h2>
            <span className="text-xs text-zinc-500">{bookmarks.length} saved locally</span>
          </div>
          {bookmarks.length ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {bookmarks.map((paper) => (
                <PaperCard key={`bookmark-${paper.id}`} paper={paper} onSelect={handleSelect} />
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-line bg-panel px-5 py-6 text-sm text-zinc-500">
              Bookmark papers from the paper preview to keep them here.
            </div>
          )}
        </section>
      </section>

      <PaperModal
        paper={selectedPaper}
        onClose={() => setSelectedPaper(null)}
        onBookmark={bookmarkPaper}
        isBookmarked={selectedPaper ? bookmarks.some((paper) => paper.id === selectedPaper.id) : false}
        onViewed={rememberPaper}
      />
    </main>
  );
}
