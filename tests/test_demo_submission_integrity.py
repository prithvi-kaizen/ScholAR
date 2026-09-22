import pytest
from pathlib import Path
from evaluation.validate_demo_paper import validate_demo_paper


def test_demo_paper_compilation_and_compliance():
    paper_dir = Path("paper/eacl_demo")
    report = validate_demo_paper(paper_dir)
    assert report["status"] == "PASS", f"Validation issues: {report.get('issues')}"
    assert report["content_page_end"] is not None
    assert report["content_page_end"] <= 6, f"Content pages exceeded 6: {report['content_page_end']}"
    assert report["claims_audited"] >= 5


def test_venue_requirements_consistency():
    req_path = Path("paper/eacl_demo/venue_requirements.json")
    assert req_path.is_file()
    import json
    req = json.loads(req_path.read_text(encoding="utf-8"))
    assert req["venue"] == "EACL 2027 System Demonstrations Track"
    assert req["review_content_pages"] == 6
    assert req["single_blind"] is True
    assert req["mandatory_video_max_minutes"] == 2.5
    assert req["mandatory_software_availability"] is True


def test_video_script_and_metadata_exist():
    assert Path("paper/eacl_demo/video_script.md").is_file()
    assert Path("paper/eacl_demo/SUBMISSION_METADATA.md").is_file()
    assert Path("paper/eacl_demo/ARTIFACT_SMOKE_TEST.md").is_file()


def test_demo_release_archive_and_sidecar():
    from scripts.package_demo_release import DEFAULT_OUTPUT, sha256_file, verify_demo_archive

    assert DEFAULT_OUTPUT.is_file(), f"Release package missing: {DEFAULT_OUTPUT}"
    sidecar = DEFAULT_OUTPUT.with_suffix(".zip.sha256")
    assert sidecar.is_file(), f"Sidecar missing: {sidecar}"

    digest = sha256_file(DEFAULT_OUTPUT)
    sidecar_content = sidecar.read_text(encoding="utf-8").strip()
    assert sidecar_content.startswith(digest)

    issues = verify_demo_archive(DEFAULT_OUTPUT)
    assert not issues, f"Archive verification issues: {issues}"


def test_demo_install_guide_integrity():
    install_guide = Path("DEMO_INSTALL.md")
    assert install_guide.is_file()
    content = install_guide.read_text(encoding="utf-8")

    assert "ScholAR_EACL2027_Demo_v1.0.0.zip" in content
    assert "make demo-setup" in content
    assert "make demo-doctor" in content
    assert "make demo-model" in content
    assert "make demo-run" in content

    # Check for disallowed placeholders
    for placeholder in ["TODO", "TBD", "FIXME", "#demo-video"]:
        assert placeholder not in content, f"Found placeholder '{placeholder}' in DEMO_INSTALL.md"
