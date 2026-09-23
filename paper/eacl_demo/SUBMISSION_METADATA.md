# EACL 2027 System Demonstrations: Submission Metadata

## Basic Information
- **Conference:** 2027 Conference of the European Chapter of the Association for Computational Linguistics (EACL 2027)
- **Track:** System Demonstrations Track
- **Submission Deadline:** September 22, 2026, 23:59 AoE
- **Paper Title:** SCHOLAR: A Local-First Research Assistant for Scientific Papers
- **Submission Format:** PDF (ACL 2027 style, A4 paper)
- **Review Policy:** Single-Blind (Author names and affiliations included)

## Authors
- **Prithviraj** (Primary Contact) — Mahindra University, CSE Dept., Hyderabad, India (`se23uari141@mahindrauniversity.edu.in`)
- **Sai Amrit Patnaik** — Avykt Ehsaas, Hyderabad, India (`saiamritp@gmail.com`)
- **Sangeeta Lamba** — Central University of Haryana (CUH), India (`sangeetalamba@cuh.ac.in`)
- **Nidhi Goyal** — Mahindra University, CSE Dept., Hyderabad, India (`nidhi.goyal@mahindrauniversity.edu.in`)

## Abstract
Scientific-document assistants can answer questions over research papers, but verifying their answers still requires readers to locate and inspect the supporting evidence. We present SCHOLAR, a local-first scientific-document assistant designed around inspectable citation grounding. SCHOLAR preserves source identity throughout document processing, allowing citations in generated answers to resolve directly to the corresponding paragraph, table, or figure region in the original PDF. The system combines lexical and dense retrieval with local answer generation, post-generation citation checks, and an auditable evidence trace. On a retrospective benchmark of 200 questions across 10 scientific papers, fused retrieval achieves a gold-page Hit@1 of 0.85 and an MRR of 0.905. Across 45 questions sampled from 9 papers, five human raters assigned a mean grounding score of 82.6%. A separate automated audit provides additional diagnostics of grounding and citation quality. SCHOLAR turns generated citations into direct, inspectable links to source evidence while keeping the question-answering workflow local. The demo and source code are available at https://github.com/prithvi-kaizen/ScholAR.

## Classification & Keywords
- **Primary Track:** System Demonstrations
- **Primary Keywords:** Retrieval-Augmented Generation, Scientific Document Question Answering, Local-First AI, Bounding-Box Grounding, Evidence Inspection, Transparent Auditing
- **Subject Areas:** Document Understanding, Information Retrieval, Natural Language Applications

## Demonstration Deliverables
1. **Manuscript PDF:** `paper/eacl_demo/main.pdf` (6 content pages + references, limitations, ethics, and appendix)
2. **Open-Source Repository:** `https://github.com/prithvi-kaizen/ScholAR.git` (MIT License)
3. **Frozen Release Package:**
   - **Release Tag:** `eacl-demo-2027-v1.0.0`
   - **Release Page:** `https://github.com/prithvi-kaizen/ScholAR/releases/tag/eacl-demo-2027-v1.0.0`
   - **Archive File:** `release/ScholAR_EACL2027_Demo_v1.0.0.zip`
   - **Package SHA-256:** `ebb3a67286c35256ccb7ca78df66371953c4f4f8bc7b791a7b1ce4f83b0e825f`
   - **Reviewer Guide:** `DEMO_INSTALL.md`
4. **Demonstration Video URL:** `https://github.com/prithvi-kaizen/ScholAR/releases/download/eacl-demo-2027-v1.0.0/ScholAR_Demo_Screencast.mp4` (Duration: $\le 2.5$ minutes)

## Compliance Declarations
- [x] The paper adheres to the single-blind review format with author details unhidden.
- [x] The main content of the paper does not exceed six pages.
- [x] Limitations and Ethical Considerations sections are included.
- [ ] The demonstration video has been uploaded and its URL is included in both the PDF and OpenReview form.
- [x] The software is open-source under a permissive license (MIT).
- [x] No private participant data or unauthorized copyright materials are distributed.

**Release reminder:** this metadata file is a checklist, not evidence that the package or video already exists. Before submission, replace both placeholders in the paper, this file, and the OpenReview form, then re-run the clean-install, link, and PDF checks.
