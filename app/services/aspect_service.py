from __future__ import annotations

from system import SentimentAnalysisSystem


def extract_main_aspect(system: SentimentAnalysisSystem, text: str) -> str:
    return system.aspect_analyzer.infer_aspect(text)


def analyze_aspects(system: SentimentAnalysisSystem, comments: list[str], model_name: str) -> dict:
    routed = system.route_by_volume(
        comments=comments,
        model_name=model_name,
        with_aspect=True,
        with_aspect_distribution=True,
        with_fine_grained=True,
        with_pain_mining=False,
    )
    return {
        "mode": routed.get("mode", "empty"),
        "aspect_distribution": routed.get("aspect_distribution", []),
        "aspect_summary": routed.get("aspect_summary", []),
        "focus_aspect_distribution": routed.get("focus_aspect_distribution", []),
    }

