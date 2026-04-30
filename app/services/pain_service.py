from __future__ import annotations

from system import SentimentAnalysisSystem


def mine_pain_points(system: SentimentAnalysisSystem, comments: list[str], model_name: str) -> dict:
    routed = system.route_by_volume(
        comments=comments,
        model_name=model_name,
        with_aspect=True,
        with_aspect_distribution=False,
        with_fine_grained=False,
        with_pain_mining=True,
    )
    return routed.get("pain_point", {})

