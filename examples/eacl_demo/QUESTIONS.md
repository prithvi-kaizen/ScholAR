# ScholAR EACL 2027 Demo: Verified Evaluation Questions

This document lists three verified evaluation questions designed to smoke-test the textual, quantitative, and visual question answering capabilities of the ScholAR system using the included sample paper (`sample_paper.pdf`).

---

### Question 1 (Textual Lookup)
- **Question:**  
  *What optimizer, learning rate, and batch size are used during training?*
- **Expected Answer:**  
  The model is trained using the **Adam optimizer** with parameters $\beta_1 = 0.9$ and $\beta_2 = 0.999$, an initial learning rate of **$2 \times 10^{-4}$**, and a mini-batch size of **32**.
- **Evidence Location:**  
  `sample_paper.pdf`, Page 1, Section 3.1 (*Experimental Setup*).
- **Target Evidence Element:**  
  Text chunk corresponding to Section 3.1 paragraph 1.

---

### Question 2 (Quantitative Table)
- **Question:**  
  *What F1 score and latency are reported for Dynamic Evidence Fusion in Table 1?*
- **Expected Answer:**  
  In Table 1, Dynamic Evidence Fusion achieves an F1 score of **88.4%** with an operational retrieval latency of **42 ms** (achieving Precision of 0.865 and Recall of 0.904).
- **Evidence Location:**  
  `sample_paper.pdf`, Page 1, Table 1 (*Comparative Evaluation of Evidence Fusion Strategies*).
- **Target Evidence Element:**  
  Table 1 display crop / row 4 bounding box.

---

### Question 3 (Visual Figure)
- **Question:**  
  *According to Figure 1, which module connects the visual feature encoder to the ranker?*
- **Expected Answer:**  
  According to Figure 1 and Section 2, the **cross-modal projection layer** connects the visual feature encoder to the evidence ranker, projecting visual crop representations into the shared retrieval embedding space.
- **Evidence Location:**  
  `sample_paper.pdf`, Page 1, Figure 1 (*Architecture of the Dynamic Evidence Fusion Pipeline*).
- **Target Evidence Element:**  
  Figure 1 image crop and caption.
