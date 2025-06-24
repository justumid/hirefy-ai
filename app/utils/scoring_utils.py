# base/scoring_utils.py

def compute_final_score(
    semantic_score: float,
    skill_overlap: float,
    psychometric_score: float,
    fairness_adjusted_score: float,
    weights: dict = None
) -> float:
    if weights is None:
        weights = {
            "semantic_score": 0.4,
            "skill_overlap": 0.3,
            "psychometric_score": 0.2,
            "fairness_adjusted_score": 0.1
        }

    final = (
        weights["semantic_score"] * semantic_score +
        weights["skill_overlap"] * skill_overlap +
        weights["psychometric_score"] * psychometric_score +
        weights["fairness_adjusted_score"] * fairness_adjusted_score
    )
    return round(final, 4)

def normalize_score(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))
