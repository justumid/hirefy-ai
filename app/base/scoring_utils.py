def compute_final_score(
    semantic_score: float,
    skill_overlap: float,
    psychometric_score: float,
    fairness_score: float,
    weights: dict = None
) -> float:
    weights = weights or {
        "semantic": 0.4,
        "skill_overlap": 0.3,
        "psychometric": 0.2,
        "fairness": 0.1,
    }
    return round(
        semantic_score * weights["semantic"] +
        skill_overlap * weights["skill_overlap"] +
        psychometric_score * weights["psychometric"] +
        fairness_score * weights["fairness"], 4
    )

def normalize_score(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))
