"""
    Metrics methods shared across notebooks
"""
try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    raisee(e)

# Kullbach-Leibler Divergence
def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P || Q) between two logit vectors, treated as categorical distributions."""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    return float(torch.sum(p * (log_p - log_q)).item())

# Cosine Similarity
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


def exact_or_keyword_match(answer: str, expected_keywords: list[str]) -> bool:
    """True if every keyword in `expected_keywords` appears in `answer` (case-insensitive)."""
    lowered = answer.lower()
    return all(keyword.lower() in lowered for keyword in expected_keywords)
