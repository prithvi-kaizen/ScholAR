"""Enforce ScholAR's acquisition/analysis network boundary.

Strict-local mode permits prepared filesystem assets and HTTP(S) services bound to
loopback addresses. It rejects acquisition before DNS resolution or client setup.
"""

from __future__ import annotations

import ipaddress
import os
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "backend" / "data" / "papers"
ENCODER_SNAPSHOTS = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
)
DEFAULT_VISUAL_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_DOCUMENT_VISUAL_MODEL = "vidore/colqwen2-v1.0-hf"


def _huggingface_snapshot_dir(model_name: str) -> Path:
    safe_model_name = model_name.strip().replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe_model_name}" / "snapshots"


class NetworkMode(str, Enum):
    ACQUISITION_ENABLED = "acquisition-enabled"
    STRICT_LOCAL = "strict-local"


class NetworkPolicyError(Exception):
    """A requested operation violates the active network boundary."""

    def __init__(self, action: str, message: str, *, url: str | None = None) -> None:
        super().__init__(message)
        self.action = action
        self.url = url
        self.code = "NETWORK_REQUIRED" if action != "local-model-endpoint" else "NON_LOOPBACK_ENDPOINT"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "action": self.action,
            "url": self.url,
            "message": str(self),
        }


class AssetStatus(BaseModel):
    asset: str
    available: bool
    detail: str


class NetworkActionStatus(BaseModel):
    action: str
    requires_external_network: bool
    allowed: bool


class NetworkPolicyStatus(BaseModel):
    mode: NetworkMode
    external_network_allowed: bool
    local_model_endpoints_only: bool
    assets: list[AssetStatus] = Field(default_factory=list)
    missing_assets: list[str] = Field(default_factory=list)
    actions: list[NetworkActionStatus] = Field(default_factory=list)


class NetworkPolicyService:
    """Central policy for all application-owned outbound network boundaries."""

    @classmethod
    def current_mode(cls) -> NetworkMode:
        raw = os.getenv("SCHOLAR_NETWORK_MODE", NetworkMode.ACQUISITION_ENABLED.value).strip().lower()
        try:
            return NetworkMode(raw)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in NetworkMode)
            raise RuntimeError(f"Invalid SCHOLAR_NETWORK_MODE={raw!r}; expected one of: {allowed}") from exc

    @classmethod
    def is_strict_local(cls) -> bool:
        return cls.current_mode() == NetworkMode.STRICT_LOCAL

    @staticmethod
    def is_loopback_url(url: str) -> bool:
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
                return False
            host = (parsed.hostname or "").rstrip(".").lower()
            if host == "localhost":
                return True
            return ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            return False

    @classmethod
    def require_local_endpoint(cls, url: str, service: str = "local model") -> None:
        if cls.is_strict_local() and not cls.is_loopback_url(url):
            raise NetworkPolicyError(
                "local-model-endpoint",
                f"Strict-local mode requires {service} to use a loopback HTTP(S) URL; got {url!r}.",
                url=url,
            )

    @classmethod
    def require_acquisition(cls, action: str, url: str | None = None) -> None:
        if cls.is_strict_local():
            raise NetworkPolicyError(
                action,
                f"{action} requires external network access and is disabled in strict-local mode. "
                "Use a prepared local asset or explicitly restart in acquisition-enabled mode.",
                url=url,
            )

    @classmethod
    def enforce_local_model_cache(cls) -> bool:
        """Set standard library offline flags and return whether cache-only loading is required."""
        if not cls.is_strict_local():
            return False
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return True

    @classmethod
    def status(cls) -> NetworkPolicyStatus:
        mode = cls.current_mode()
        prepared_papers = any(
            path.is_dir()
            and (path / "paper.pdf").is_file()
            and (path / "chunks.json").is_file()
            and (path / "metadata.json").is_file()
            for path in PAPERS_DIR.glob("*")
        ) if PAPERS_DIR.is_dir() else False
        encoder_cached = ENCODER_SNAPSHOTS.is_dir() and any(ENCODER_SNAPSHOTS.iterdir())
        visual_model = os.getenv(
            "SCHOLAR_VISUAL_EMBEDDING_MODEL",
            DEFAULT_VISUAL_EMBEDDING_MODEL,
        ).strip() or DEFAULT_VISUAL_EMBEDDING_MODEL
        visual_snapshots = _huggingface_snapshot_dir(visual_model)
        visual_encoder_cached = visual_snapshots.is_dir() and any(visual_snapshots.iterdir())
        document_visual_model = os.getenv(
            "SCHOLAR_DOCUMENT_VISUAL_MODEL",
            DEFAULT_DOCUMENT_VISUAL_MODEL,
        ).strip() or DEFAULT_DOCUMENT_VISUAL_MODEL
        document_visual_snapshots = _huggingface_snapshot_dir(document_visual_model)
        document_visual_cached = (
            document_visual_snapshots.is_dir()
            and any(document_visual_snapshots.iterdir())
        )
        docling_artifacts = os.getenv("DOCLING_ARTIFACTS_PATH", "").strip()
        parser_assets = bool(docling_artifacts and Path(docling_artifacts).expanduser().is_dir())
        assets = [
            AssetStatus(
                asset="prepared_papers",
                available=prepared_papers,
                detail="At least one canonical local paper bundle is available." if prepared_papers else "No canonical prepared paper bundle was found.",
            ),
            AssetStatus(
                asset="embedding_model",
                available=encoder_cached,
                detail="Cached all-MiniLM-L6-v2 snapshot found." if encoder_cached else "Dense retrieval will use its labeled deterministic fallback unless a snapshot is acquired separately.",
            ),
            AssetStatus(
                asset="visual_embedding_model",
                available=visual_encoder_cached,
                detail=(
                    f"Cached paired image/text snapshot found for {visual_model}."
                    if visual_encoder_cached
                    else f"Always-on image retrieval is disabled until {visual_model} is acquired separately."
                ),
            ),
            AssetStatus(
                asset="document_visual_retrieval_model",
                available=document_visual_cached,
                detail=(
                    f"Cached document-visual snapshot found for {document_visual_model}."
                    if document_visual_cached
                    else (
                        f"The ColQwen2 page retriever is unavailable until "
                        f"{document_visual_model} is acquired separately; auto mode may use CLIP."
                    )
                ),
            ),
            AssetStatus(
                asset="docling_artifacts",
                available=parser_assets,
                detail=(f"Configured at {docling_artifacts}." if parser_assets else "DOCLING_ARTIFACTS_PATH is not configured; strict-local ingestion uses the PyMuPDF fallback."),
            ),
        ]
        external_allowed = mode == NetworkMode.ACQUISITION_ENABLED
        actions = [
            NetworkActionStatus(action="analyze-prepared-paper", requires_external_network=False, allowed=True),
            NetworkActionStatus(action="loopback-model-inference", requires_external_network=False, allowed=True),
            NetworkActionStatus(action="upload-local-pdf", requires_external_network=False, allowed=True),
            NetworkActionStatus(action="search-arxiv", requires_external_network=True, allowed=external_allowed),
            NetworkActionStatus(action="download-paper", requires_external_network=True, allowed=external_allowed),
            NetworkActionStatus(action="resolve-references", requires_external_network=True, allowed=external_allowed),
            NetworkActionStatus(action="download-reference", requires_external_network=True, allowed=external_allowed),
            NetworkActionStatus(action="acquire-model-or-dataset", requires_external_network=True, allowed=external_allowed),
        ]
        return NetworkPolicyStatus(
            mode=mode,
            external_network_allowed=external_allowed,
            local_model_endpoints_only=mode == NetworkMode.STRICT_LOCAL,
            assets=assets,
            missing_assets=[asset.asset for asset in assets if not asset.available],
            actions=actions,
        )
