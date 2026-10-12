"""Built-in toy datasets used by the notebooks, so the notebooks themselves
stay focused on the ablation logic rather than data wrangling.

All data in this module is LLM-generated, using Claude Code.
"""

# (prompt, correct_completion, incorrect_completion) — single-word completions,
# used for logit-diff style factual-recall ablation sweeps.
FACTUAL_RECALL_PAIRS: list[tuple[str, str, str]] = [
    ("The capital of France is", "Paris", "London"),
    ("The capital of Japan is", "Tokyo", "Seoul"),
    ("The capital of Italy is", "Rome", "Madrid"),
    ("The capital of Germany is", "Berlin", "Vienna"),
    ("The capital of Egypt is", "Cairo", "Baghdad")
]
GENERAL_KNOWLEDGE_RECALL_PAIRS: list[tuple[str, str, str]] = [
    ("The name of the matemathician Cantor was", "Georg", "Hans"),
    ("A black hole is the remnant of a dead", "star", "galaxy"),
    ("The sunset sky is", "red", "green")
]

# (prompt, correct_name, incorrect_name) — IOI-style "indirect object" templates.
# Exploratory: the classic clean circuit was found on GPT-2-small, not a small
# instruction-tuned model, so treat whatever shows up here as a starting point
# for investigation rather than a guaranteed reproduction.
IOI_STYLE_TEMPLATES: list[tuple[str, str, str]] = [
    ("When Mary and John went to the store, John gave a drink to", "Mary", "John"),
    ("When Alice and Bob went to the park, Bob gave the ball to", "Alice", "Bob"),
    ("After Tom and Sam left the office, Sam handed the keys to", "Tom", "Sam"),
    ("When Emma and Liam were at the cafe, Liam passed the menu to", "Emma", "Liam"),
]