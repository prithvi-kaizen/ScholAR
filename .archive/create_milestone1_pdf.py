from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "ScholAR_Milestone_1.pdf"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.75 * inch, 0.45 * inch, "ScholAR Milestone 1")
    canvas.drawRightString(7.75 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=colors.HexColor("#111111"),
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverSub",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#333333"),
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#111111"),
        spaceBefore=12,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="BodySimple",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#222222"),
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="BulletSimple",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=15,
        leftIndent=16,
        firstLineIndent=-8,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#444444"),
    )
)


def p(text):
    return Paragraph(text, styles["BodySimple"])


def h(text):
    return Paragraph(text, styles["Section"])


def bullet(text):
    return Paragraph(f"- {text}", styles["BulletSimple"])


def cover():
    return [
        Spacer(1, 1.2 * inch),
        Paragraph("ScholAR", styles["CoverTitle"]),
        Paragraph("Local LLM Research Paper Assistant", styles["CoverSub"]),
        Paragraph("Milestone 1 Report", styles["CoverSub"]),
        Paragraph("April 30, 2026", styles["CoverSub"]),
        Spacer(1, 0.5 * inch),
        Table(
            [
                ["Project goal", "Search, process, view, and study arXiv papers with a local Qwen model."],
                ["Milestone focus", "A simple working MVP with local storage, PDF processing, study goals, and grounded chat."],
                ["Status", "Core flow implemented and tested with Attention Is All You Need."],
            ],
            colWidths=[1.5 * inch, 4.8 * inch],
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F7F9")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D8DCE2")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8DCE2")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ],
        ),
        PageBreak(),
    ]


story = []
story.extend(cover())

story.extend(
    [
        h("1. Project Overview"),
        p(
            "ScholAR is a local research paper assistant for students, engineers, and researchers. "
            "The goal is to make reading papers less stressful and more structured. A user can search "
            "for papers from arXiv, select one paper, download and process the PDF, view the paper, "
            "and study it with help from a local Ollama Qwen model."
        ),
        p(
            "This milestone focuses on a simple working system. The project does not include login, "
            "deployment, recommendations, fine-tuning, or a knowledge graph yet."
        ),
        h("2. Real Problem"),
        p(
            "Research papers are difficult to read because they are long, dense, and often assume "
            "background knowledge. Many readers understand the abstract but struggle to connect the "
            "method, experiments, results, and implementation details."
        ),
        p(
            "This affects students, applied AI engineers, independent researchers, and people who need "
            "to study new papers quickly. It is especially hard when a person is reading outside their "
            "main area of expertise."
        ),
        h("3. Why Existing Solutions Fall Short"),
        p(
            "Normal PDF readers show the paper, but they do not guide learning. General chatbots can "
            "summarize text, but they may send private documents to cloud services, lose source grounding, "
            "or answer in a way that is not clearly tied to the paper."
        ),
        p(
            "Academic search tools are useful for finding papers, but they usually stop at metadata, "
            "citations, and abstracts. They do not turn a selected paper into a study session."
        ),
        h("4. Proposed Approach"),
        p("ScholAR combines four simple parts:"),
        bullet("arXiv search to find papers."),
        bullet("Local PDF download and text extraction."),
        bullet("Simple chunking and keyword retrieval."),
        bullet("Local Qwen through Ollama for study goals and paper chat."),
        p(
            "The assistant answers questions using retrieved paper chunks and returns page citations. "
            "If the local model is unavailable or too slow, the system still returns fallback study goals "
            "and grounded extractive answers."
        ),
        h("5. Milestone 1 Scope"),
        p("Milestone 1 includes the core end-to-end workflow:"),
    ]
)

for item in [
    "Search arXiv papers from the home page.",
    "Open a paper card and view paper details.",
    "Prepare the paper by downloading the PDF.",
    "Extract text page by page with PyMuPDF.",
    "Save metadata, pages, and chunks locally.",
    "View the full paper in a side-by-side study workspace.",
    "Generate 8 study goals.",
    "Ask questions and receive answers with page citations.",
]:
    story.append(bullet(item))

story.extend(
    [
        h("6. System Architecture"),
        p("Frontend:"),
        bullet("Next.js with TypeScript."),
        bullet("Tailwind CSS."),
        bullet("Dark study interface."),
        bullet("Components for search, paper cards, modal, PDF viewer, study goals, and chat."),
        p("Backend:"),
        bullet("FastAPI."),
        bullet("arXiv Atom API for paper search."),
        bullet("PyMuPDF for PDF text extraction and page rendering."),
        bullet("Ollama for local LLM calls."),
        bullet("Local file storage under backend/data/papers."),
        p("Local model:"),
        bullet("Ollama running locally."),
        bullet("Current model used on this machine: qwen3.5:9b."),
        bullet("Default environment variable support: OLLAMA_MODEL."),
        h("7. Data Pipeline"),
        p("The pipeline works as follows:"),
    ]
)

for item in [
    "The user searches arXiv.",
    "The backend returns cleaned paper metadata.",
    "The user clicks Study with AI.",
    "The backend creates a safe local paper folder.",
    "The PDF is downloaded as paper.pdf.",
    "PyMuPDF extracts page-wise text.",
    "The system saves metadata.json, pages.json, and chunks.json.",
    "The PDF pages are rendered as PNG images for reliable viewing in the browser.",
]:
    story.append(bullet(item))

story.extend(
    [
        p(
            "The chunking method is simple. Each page is split into chunks of about 1000 to 1800 words "
            "while preserving page numbers."
        ),
        h("8. Model and Retrieval"),
        p(
            "The current assistant uses keyword overlap scoring for retrieval. When a user asks a question, "
            "the backend loads chunks.json, scores chunks against the question, and selects the top 4 chunks."
        ),
        p(
            "The selected chunks are placed into a grounded prompt for Ollama. The model is instructed to "
            "answer only from the paper context and cite page numbers such as [p. 2]."
        ),
        p(
            "This is not a full RAG system with embeddings yet. That is acceptable for Milestone 1 because "
            "the retrieval method is transparent and easy to debug."
        ),
        h("9. User Interface"),
        p("The UI has two main views:"),
        p("Home page:"),
        bullet("Search bar."),
        bullet("Paper cards."),
        bullet("Paper details modal."),
        bullet("Recently viewed papers."),
        bullet("Local bookmarks."),
        p("Study page:"),
        bullet("Full paper viewer on the left."),
        bullet("AI study panel on the right."),
        bullet("Study goals tab."),
        bullet("Quick start tab."),
        bullet("Chat box at the bottom."),
        p("The study page now shows the complete paper as scrollable rendered pages, not only the first page."),
        PageBreak(),
        h("10. Preliminary Results"),
    ]
)

table_data = [
    ["Test paper", "Pages", "Chunks", "Study goals", "Local answer"],
    ["Attention Is All You Need", "15", "15", "8 goals shown", "Answer returned with page citations"],
    ["RAG survey", "TBD", "TBD", "Pending", "Pending"],
]
story.append(
    Table(
        table_data,
        colWidths=[2.0 * inch, 0.7 * inch, 0.7 * inch, 1.35 * inch, 1.7 * inch],
        repeatRows=1,
        style=[
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9CED6")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F8FA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ],
    )
)

story.extend(
    [
        Spacer(1, 0.15 * inch),
        p(
            "The first test confirms that the full search to study flow works: search arXiv, prepare paper, "
            "render PDF pages, load study goals, ask a question, and receive a grounded answer."
        ),
        h("11. Known Limitations"),
    ]
)

for item in [
    "Retrieval is keyword based, not embedding based.",
    "PDF extraction depends on the quality of the PDF text layer.",
    "Ollama can be slow on CPU or when the model is large.",
    "Study goals currently use fallback goals if the model takes too long.",
    "Local storage is single-user and file based.",
    "There is no deployment or authentication in this milestone.",
]:
    story.append(bullet(item))

story.extend([h("12. Next Steps")])

for item in [
    "Add embedding based retrieval.",
    "Improve citation matching.",
    "Add better model timeout and streaming behavior.",
    "Add evaluation questions for study goals.",
    "Add a small results log for more test papers.",
    "Improve PDF page navigation and search.",
]:
    story.append(bullet(item))

story.extend(
    [
        h("13. Conclusion"),
        p(
            "Milestone 1 successfully builds a working local paper study assistant. The system is small, "
            "understandable, and useful enough to test with real arXiv papers. It proves the main workflow "
            "before adding more advanced research features."
        ),
    ]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc = BaseDocTemplate(
    str(OUTPUT),
    pagesize=letter,
    rightMargin=0.75 * inch,
    leftMargin=0.75 * inch,
    topMargin=0.7 * inch,
    bottomMargin=0.65 * inch,
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
doc.build(story)
print(OUTPUT)
