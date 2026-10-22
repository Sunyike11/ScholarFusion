PAPER_CARD_PROMPT_TEMPLATE = """
Extract a conservative structured paper card from the provided evidence bundle.
Use unknown when evidence is insufficient. Every non-unknown field must cite evidence.
""".strip()


COMPARISON_PROMPT_TEMPLATE = """
Compare papers using the fixed research schema. Do not fill missing information.
Distinguish explicit facts, system summaries, and missing information.
""".strip()
