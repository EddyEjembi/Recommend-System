"""Persona-generation prompt strings (Phase 4 LLM)."""

PERSONA_SYSTEM_PROMPT = """You design realistic reviewer personas for a behavioral review simulation system.

Output exactly ONE JSON object (no markdown, no extra text).

Required keys (exact spelling):
- "user_id": string (use the id from the user message)
- "voice": string — tone, habits, and writing style in one paragraph
- "preferences": array of strings — short factors the user cares about (NOT a nested object)
- "dealbreakers": array of strings — things that annoy them or ruin a visit
- "typical_length": integer — average review length in characters
- "vocabulary_quirks": array of strings — recurring language habits (empty array if none)
- "metadata": object — optional extras such as emotional_style, rating_behavior, review_priorities, archetype

Example shape:
{"user_id":"...", "voice":"...", "preferences":["likes generous portions","mid budget"], "dealbreakers":["slow service"], "typical_length":400, "vocabulary_quirks":["occasional Yoruba phrases"], "metadata":{"emotional_style":"warm and practical"}}

Ground everything in the supplied evidence. If evidence is weak, stay conservative.
Nigerian English or mild pidgin only when supported by the seed or neighbours — never forced or exaggerated."""

PERSONA_WARM_USER_TEMPLATE = """Build a persona JSON for this user.

user_id: {user_id}

Deterministic behaviour profile (from offline analysis — trust these numbers):
{behavior_json}

Their recent review excerpts (verbatim; do not invent facts not supported here):
{reviews_block}

Respond with the required JSON object only."""

PERSONA_COLD_USER_TEMPLATE = """Build a persona JSON for a **cold-start** user (no Yelp history in our dataset).

Target user_id: {user_id}

Seed profile (authoritative — this is who they are):
{cold_seed_json}

Vector-database neighbours (similar reviewers in embedding space). Use their behaviour summaries only to infer **writing style overlap** and plausible habits — the target identity stays the seed above, not a clone of a neighbour:
{neighbors_block}

Respond with the required JSON object only."""
