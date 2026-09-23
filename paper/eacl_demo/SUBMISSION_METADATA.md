# EACL 2027 System Demonstrations: Submission Metadata

## Basic Information
- **Conference:** 2027 Conference of the European Chapter of the Association for Computational Linguistics (EACL 2027)
- **Track:** System Demonstrations Track
- **Submission Deadline:** September 22, 2026, 23:59 AoE
- **Paper Title:** ScholAR: Inspectable Local-First Question Answering for Scientific Papers
- **Submission Format:** PDF (ACL 2027 style, A4 paper)
- **Review Policy:** Single-Blind (Author names and affiliations included)

## Authors
- **Primary Author:** Prithviraj Patil
- **Affiliation:** Independent Researcher
- **Email:** `prithvisp28@gmail.com`
- **GitHub:** `@prithvi-kaizen`

## Abstract
Researchers often need to verify a paper claim against the exact paragraph, table, or figure that supports it. ScholAR is a local-first scientific-document assistant built around that interaction: a user asks a question, receives a cited answer, and clicks a citation to inspect the linked source region in the original PDF. The system preserves page identity and layout coordinates during ingestion, combines lexical and dense retrieval, and records the evidence and verifier actions behind each response. In a retrospective benchmark of 100 questions across 25 papers, fused retrieval reaches gold-page Hit@1 of 0.85; a separate 150-case language-model-based audit reports grounding and citation categories, but is not human ground truth. The demonstration shows how researchers, students, and reviewers can use this inspectable workflow locally, with the final submission providing an installable package and screencast link.

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
