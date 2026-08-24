#!/usr/bin/env python3
"""Batch downloader and ingestion pipeline for the 10 requested test papers."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.chunking_service import chunk_pages
from backend.services.pdf_service import (
    extract_figures,
    extract_pages,
    paper_dir,
    safe_paper_id,
    write_json,
    read_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_papers")

PAPERS_DATA_DIR = ROOT / "backend" / "data" / "papers"
PAPERS_DATA_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_TO_INGEST = [
    {
        "id": "1406.2661",
        "title": "Generative Adversarial Nets (GAN)",
        "authors": ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza", "Bing Xu", "David Warde-Farley", "Sherjil Ozair", "Aaron Courville", "Yoshua Bengio"],
        "year": 2014,
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
        "urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf",
            "https://arxiv.org/pdf/1406.2661.pdf",
            "https://export.arxiv.org/pdf/1406.2661.pdf",
        ],
    },
    {
        "id": "1412.6980",
        "title": "Adam: A Method for Stochastic Optimization",
        "authors": ["Diederik P. Kingma", "Jimmy Ba"],
        "year": 2014,
        "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.",
        "urls": [
            "https://arxiv.org/pdf/1412.6980.pdf",
            "https://export.arxiv.org/pdf/1412.6980.pdf",
        ],
    },
    {
        "id": "2112.10752",
        "title": "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)",
        "authors": ["Robin Rombach", "Andreas Blattmann", "Dominik Lorenz", "Patrick Esser", "Björn Ommer"],
        "year": 2021,
        "abstract": "By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models achieve state-of-the-art synthesis results on image data and beyond. We apply them in the latent space of powerful pretrained autoencoders.",
        "urls": [
            "https://arxiv.org/pdf/2112.10752.pdf",
            "https://export.arxiv.org/pdf/2112.10752.pdf",
        ],
    },
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need (Transformer)",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
        "year": 2017,
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.",
        "urls": [
            "https://arxiv.org/pdf/1706.03762.pdf",
            "https://export.arxiv.org/pdf/1706.03762.pdf",
        ],
    },
    {
        "id": "2406.08394",
        "title": "VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model",
        "authors": ["Jiannan Wu", "Zheng Ma", "Zhenyu Yang", "Zhaoyang Zeng", "Yuechen Zhang", "Hao Zhang", "Feng Li", "Tianhe Ren", "Liunian Harold Li", "Bo Dai", "Lei Zhang"],
        "year": 2024,
        "abstract": "We present VisionLLM v2, an end-to-end generalist multimodal large language model that unifies visual perception, understanding, and generation in a single framework with interactive visual anchor tokens.",
        "urls": [
            "https://arxiv.org/pdf/2406.08394.pdf",
            "https://export.arxiv.org/pdf/2406.08394.pdf",
        ],
    },
    {
        "id": "2104.08663",
        "title": "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
        "authors": ["Nandan Thakur", "Nils Reimers", "Andreas Rücklé", "Abhishek Srivastava", "Iryna Gurevych"],
        "year": 2021,
        "abstract": "We introduce BEIR, a heterogeneous benchmark containing 18 diverse datasets across 9 distinct information retrieval tasks to evaluate the zero-shot generalization capabilities of neural search models.",
        "urls": [
            "https://arxiv.org/pdf/2104.08663.pdf",
            "https://export.arxiv.org/pdf/2104.08663.pdf",
        ],
    },
    {
        "id": "2603.14257",
        "title": "Automatic Inter-document Multi-hop Scientific QA Generation",
        "authors": ["Seungmin Lee", "Dongha Kim", "Yuni Jeon", "Junyoung Koh", "Min Song"],
        "year": 2026,
        "abstract": "We propose AIM-SciQA, an automated framework to generate multi-document, multi-hop scientific QA datasets leveraging LLMs and cross-document citation embedding alignment.",
        "urls": [
            "https://arxiv.org/pdf/2603.14257.pdf",
            "https://export.arxiv.org/pdf/2603.14257.pdf",
        ],
    },
    {
        "id": "2025.emnlp-main.77",
        "title": "MEBench: Benchmarking Large Language Models for Cross-Document Multi-Entity Question Answering",
        "authors": ["Yao Zhang", "Sheng Shen", "Dan Roth", "Heng Ji"],
        "year": 2025,
        "abstract": "We introduce MEBench, a comprehensive benchmark evaluating LLMs on cross-document multi-entity question answering across comparative, statistical, and relational reasoning tasks.",
        "urls": [
            "https://aclanthology.org/2025.emnlp-main.77.pdf",
            "https://arxiv.org/pdf/2407.03741.pdf",
            "https://export.arxiv.org/pdf/2407.03741.pdf",
        ],
    },
    {
        "id": "yale_thesis_1003",
        "title": "Towards Multi-Modal Multi-Document Understanding Capabilities in Foundation Models (M3SciQA)",
        "authors": ["Chuhan Li", "Alexander R. Fabbri", "Arman Cohan"],
        "year": 2024,
        "abstract": "This thesis explores multi-modal and multi-document scientific representation, cross-document reasoning, and retrieval-augmented generation architectures for foundation models.",
        "urls": [
            "https://arxiv.org/pdf/2406.03716.pdf",
            "https://export.arxiv.org/pdf/2406.03716.pdf",
            "https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1003&context=computer_science_theses",
        ],
    },
    {
        "id": "2410.00526",
        "title": "Conversational QA in Multi-instructional Documents",
        "authors": ["Yixuan Su", "Kexin Wang", "Zuchao Li", "Linfeng Song"],
        "year": 2024,
        "abstract": "We formulate and benchmark conversational question answering in multi-instructional documents, studying dialogue state tracking and context-aware evidence retrieval.",
        "urls": [
            "https://arxiv.org/pdf/2410.00526.pdf",
            "https://export.arxiv.org/pdf/2410.00526.pdf",
        ],
    },
]


def download_pdf(urls: list[str], dest_path: Path) -> bool:
    """Download PDF from list of candidate URLs with user-agent headers, SSL fallback, and curl."""
    import ssl
    import subprocess
    import shutil

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 ScholAR/1.0",
        "Accept": "application/pdf,*/*",
    }
    ctx = ssl._create_unverified_context()

    for url in urls:
        logger.info(f"Attempting download from: {url}")
        try:
            req = Request(url, headers=headers)
            with urlopen(req, context=ctx, timeout=30) as resp:
                data = resp.read()
                if data[:4] == b"%PDF" or b"%PDF" in data[:1024]:
                    dest_path.write_bytes(data)
                    logger.info(f"Successfully downloaded {len(data) / 1024 / 1024:.2f} MB from {url}")
                    return True
                else:
                    logger.warning(f"Downloaded content from {url} is not a valid PDF (size: {len(data)} bytes).")
        except Exception as e:
            logger.warning(f"urllib download failed for {url}: {e}")

        # Fallback to curl
        if shutil.which("curl"):
            try:
                logger.info(f"Trying curl for {url}...")
                subprocess.run(
                    [
                        "curl",
                        "-sSL",
                        "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                        url,
                        "-o", str(dest_path),
                    ],
                    check=True,
                    timeout=45,
                )
                if dest_path.exists() and dest_path.stat().st_size > 1000:
                    with open(dest_path, "rb") as f:
                        magic = f.read(1024)
                    if b"%PDF" in magic:
                        logger.info(f"curl successfully downloaded {dest_path.stat().st_size / 1024 / 1024:.2f} MB from {url}")
                        return True
                    else:
                        dest_path.unlink(missing_ok=True)
            except Exception as ce:
                logger.warning(f"curl failed for {url}: {ce}")

        time.sleep(1)
    return False


def ingest_paper(paper_meta: dict[str, Any]) -> bool:
    """Fully ingest a single paper into ScholAR storage."""
    paper_id = safe_paper_id(paper_meta["id"])
    target_dir = paper_dir(paper_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / "paper.pdf"

    pages_path = target_dir / "pages.json"
    chunks_path = target_dir / "chunks.json"
    figures_path = target_dir / "figures.json"
    figures_dir = target_dir / "figures"
    meta_path = target_dir / "metadata.json"

    # Download PDF if not present
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        success = download_pdf(paper_meta["urls"], pdf_path)
        if not success:
            logger.error(f"Could not download PDF for {paper_meta['title']}")
            return False
    else:
        logger.info(f"PDF already exists for [{paper_id}]")

    logger.info(f"Extracting pages for [{paper_id}] {paper_meta['title']}...")
    try:
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        doc.close()

        if not pages_path.exists():
            pages = extract_pages(pdf_path)
            write_json(pages_path, pages)
        else:
            pages = read_json(pages_path)

        if not chunks_path.exists():
            logger.info(f"Chunking pages for [{paper_id}]...")
            chunks = chunk_pages(pages)
            write_json(chunks_path, chunks)
        else:
            chunks = read_json(chunks_path)

        if not figures_path.exists():
            logger.info(f"Extracting figures and tables for [{paper_id}]...")
            figures = extract_figures(pdf_path, figures_dir)
            write_json(figures_path, figures)
        else:
            figures = read_json(figures_path)

        # Write metadata.json
        meta_payload = {
            "id": paper_id,
            "title": paper_meta["title"],
            "authors": paper_meta["authors"],
            "year": paper_meta["year"],
            "abstract": paper_meta["abstract"],
            "pdf_url": paper_meta["urls"][0],
            "page_count": num_pages,
            "chunk_count": len(chunks),
            "figure_count": len(figures),
            "ingested_at": time.time(),
        }
        write_json(meta_path, meta_payload)
        logger.info(f"Successfully ingested [{paper_id}]: {num_pages} pages, {len(chunks)} chunks, {len(figures)} figures.")
        return True

    except Exception as exc:
        logger.error(f"Error processing {paper_id}: {exc}", exc_info=True)
        return False


def main():
    logger.info(f"Starting ingestion of {len(PAPERS_TO_INGEST)} papers...")
    results = {}
    for idx, paper in enumerate(PAPERS_TO_INGEST, 1):
        logger.info(f"\n[{idx}/{len(PAPERS_TO_INGEST)}] Processing {paper['title']} (ID: {paper['id']})...")
        success = ingest_paper(paper)
        results[paper["id"]] = {"title": paper["title"], "success": success}
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("Ingestion Summary:")
    print("=" * 60)
    for pid, res in results.items():
        status = "LOADED" if res["success"] else "FAILED"
        print(f"  [{status}] {pid}: {res['title']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
