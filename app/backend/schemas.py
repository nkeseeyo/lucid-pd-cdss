"""Pydantic request/response models for the NeuroVox CDSS API.

Shape matches what the React UI renders: SHAP features carry a relative weight and
a direction (raises/lowers risk); the recommendation may name a secondary specialist.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str


class Feature(BaseModel):
    name: str
    weight: float          # 0..1 relative SHAP contribution (bar length)
    dir: str               # "up" raises risk, "down" lowers it


class Explanation(BaseModel):
    features: list[Feature] = Field(default_factory=list)  # empty for MRI (uses heatmap)
    plain_text: str = ""
    method: str = "SHAP"   # or "Grad-CAM"


class Recommendation(BaseModel):
    band: str
    route: str
    specialist_type: str
    secondary: str = ""
    disclaimer: str


class CombinedResult(BaseModel):
    modality: str          # "voice" | "mri" | "combined"
    probability: float     # model PD probability, 0..1
    risk_band: str         # low | moderate | high
    explanation: Explanation
    recommendation: Recommendation
    caveat: str = ""       # MRI critique-baseline / illustrative-fusion note
    gradcam: str = ""      # data-URL PNG of the Grad-CAM overlay (MRI/combined only)
