from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from backend.services.network_policy_service import NetworkPolicyService


class CapabilityMode(str, Enum):
    AUTO = "AUTO"
    TEXT_ONLY = "TEXT_ONLY"
    NATIVE_VISION = "NATIVE_VISION"
    RESEARCH_CONTROLLED = "RESEARCH_CONTROLLED"


class HardwareTier(str, Enum):
    TIER_8GB = "8GB"
    TIER_16GB = "16GB"
    TIER_32GB_PLUS = "32GB+"


class EvidenceBudget(BaseModel):
    """Dynamic evidence and token budget tailored to consumer hardware tier."""
    hardware_tier: HardwareTier = HardwareTier.TIER_16GB
    max_context_tokens: int = 4096
    max_evidence_blocks: int = 6
    max_table_blocks: int = 2
    max_visual_crops: int = 1
    allow_vision_pixels: bool = True


class ModelCapabilities(BaseModel):
    """Normalized model capability profile for local generators and VLMs."""
    model_id: str
    display_name: str
    backend: str = "ollama"  # "ollama" | "transformers" | "vllm" | "llamacpp"
    quantization: str | None = None
    supports_text: bool = True
    supports_vision: bool = False
    supports_multi_image: bool = False
    supports_json_schema: bool = True
    context_length: int = 16000
    max_images_per_request: int | None = None
    available_ram_gb: float | None = None
    available_vram_gb: float | None = None
    capability_mode: CapabilityMode = CapabilityMode.AUTO
    active: bool = True

    def can_process_images(self) -> bool:
        """True only if the active mode and model support visual pixel inputs."""
        if self.capability_mode == CapabilityMode.TEXT_ONLY:
            return False
        return bool(self.supports_vision)

    def effective_image_limit(self) -> int:
        """Maximum images permitted for a single prompt."""
        if not self.can_process_images():
            return 0
        return self.max_images_per_request or 4


# Registry of known model architectures and capability signatures
_KNOWN_VISION_PATTERNS = (
    "qwen3.5",
    "qwen3-vl",
    "qwen2.5-vl",
    "qwen-vl",
    "gemma4",
    "llava",
    "minicpm-v",
    "llama-3.2-11b-vision",
    "llama-3.2-90b-vision",
    "mistral-large-3",
    "pixtral",
)


class ModelRegistry:
    """Registry managing model discovery and capability resolution."""
    _custom_registry: dict[str, ModelCapabilities] = {}

    @classmethod
    def register_model(cls, capabilities: ModelCapabilities) -> None:
        cls._custom_registry[capabilities.model_id.lower()] = capabilities

    @classmethod
    def resolve_capabilities(
        cls,
        model_id: str,
        mode: CapabilityMode = CapabilityMode.AUTO,
    ) -> ModelCapabilities:
        normalized_id = model_id.strip().lower()
        if normalized_id in cls._custom_registry:
            custom = cls._custom_registry[normalized_id].model_copy()
            if mode != CapabilityMode.AUTO:
                custom.capability_mode = mode
            return custom

        # Infer native vision capability from model identity
        is_vision = any(pattern in normalized_id for pattern in _KNOWN_VISION_PATTERNS)
        
        display = model_id.split("/")[-1].replace(":", " ").title()

        resolved = ModelCapabilities(
            model_id=model_id,
            display_name=display,
            backend="ollama",
            supports_text=True,
            supports_vision=is_vision,
            supports_multi_image=is_vision,
            supports_json_schema=True,
            context_length=16000 if "16k" not in normalized_id else 16000,
            max_images_per_request=4 if is_vision else None,
            capability_mode=mode,
        )
        return resolved

    @classmethod
    def list_known_models(cls) -> list[ModelCapabilities]:
        defaults = [
            ModelCapabilities(
                model_id="qwen3.5:9b",
                display_name="Qwen 3.5 (9B) - Native Multimodal",
                backend="ollama",
                supports_vision=True,
                supports_multi_image=True,
                max_images_per_request=4,
            ),
            ModelCapabilities(
                model_id="gemma4:12b",
                display_name="Gemma 4 (12B) - Multimodal",
                backend="ollama",
                supports_vision=True,
                supports_multi_image=True,
                max_images_per_request=4,
            ),
            ModelCapabilities(
                model_id="llama3.1:8b",
                display_name="Llama 3.1 (8B) - Text Only",
                backend="ollama",
                supports_vision=False,
                supports_multi_image=False,
            ),
            ModelCapabilities(
                model_id="mistral:7b",
                display_name="Mistral (7B) - Text Only",
                backend="ollama",
                supports_vision=False,
                supports_multi_image=False,
            ),
        ]
        return list(cls._custom_registry.values()) or defaults

    @classmethod
    async def discover_ollama_models(cls, base_url: str = "http://localhost:11434") -> list[ModelCapabilities]:
        """Query a policy-approved Ollama server and register installed models."""
        import httpx
        NetworkPolicyService.require_local_endpoint(base_url, "Ollama")
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                response = await client.get(f"{base_url.rstrip('/')}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("models", []):
                        name = item.get("name")
                        if name:
                            resolved = cls.resolve_capabilities(name)
                            cls.register_model(resolved)
        except Exception:
            pass
        return cls.list_known_models()
