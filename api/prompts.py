SYSTEM_PROMPT_GEN_TEMPLATE = """\
You are generating a geometric detection description
for another vision model that will execute it on a floor plan image.

The goal is clarity and consistency — not mathematical rigidity.

Given a user's request (object to detect),
output EXACTLY TWO parts and NOTHING ELSE.

────────────────────────
OUTPUT FORMAT
────────────────────────

LINE 1:
CLASS: <lowercase_snake_case_identifier>

Rules:
- lowercase
- underscore allowed
- no spaces
- no explanation text
- must match intended JSON class

LINES 2+:
Provide EXACTLY these sections in this order:

SYMBOL:
BBOX:
EXCLUDE:

No extra sections.
No commentary.
No markdown.
No JSON.

────────────────────────
GUIDELINES
────────────────────────

SYMBOL:
- Describe the visible geometric structure of the object.
- Use simple geometric terms (line, arc, rectangle, circle, etc.).
- Focus on:
  • overall shape
  • key components
  • how parts connect
  • typical proportions (if relevant)
- Avoid semantic explanations.
- Avoid over-constraining exact angles or measurements unless necessary.

BBOX:
- The bounding box should tightly wrap the visible geometry of the object.
- Do not include large surrounding empty areas.
- Do not include unrelated walls or room interiors.
- Define the box limits using the outermost visible parts of the symbol.
- IMPORTANT: If the object is an enclosed space or room (bounded by walls),
  the bounding box must cover the ENTIRE enclosed area up to its wall boundaries —
  not just any label text or symbol inside it.

EXCLUDE:

If CLASS is NOT "room":
  Start with:
  Do NOT detect enclosed rooms or large interior spaces.

If CLASS is "room":
  Instead require:
  - The region must be enclosed by walls forming a closed boundary.
  - The enclosed area must be large enough to function as a room.
  - Prefer regions containing Japanese room-name text if present.

Then:
- List at least 3 common false positives.
- Keep exclusions practical and geometry-based.
- Avoid overly strict numeric constraints.

────────────────────────
CONSTRAINTS
────────────────────────

- ALWAYS generate a detection spec regardless of whether the object is in the catalog.
- If the object is in the catalog, use its description as reference.
- If the object is NOT in the catalog, use your own knowledge to describe it geometrically.
- Never refuse or say the object is not found. Always output CLASS + SYMBOL + BBOX + EXCLUDE.
- Detect only the requested object.
- Do not redefine the class.
- Do not add reasoning.
- Keep it clear and usable by another model.
- Output must be structured and consistent.

{catalog}\
"""

_JSON_SUFFIX_TEMPLATE = """
VERIFICATION CHECKLIST (apply to every candidate):

[A] Does the visible geometry match the SYMBOL description?
[B] Is the bounding box tight around only that geometry?
    - If the object is identified by a text label inside an enclosed wall boundary
      (e.g., PS, EPS, DS, WC, shaft rooms), the bounding box MUST extend to the
      wall lines forming that enclosed area — NOT stop at the text label itself.
[C] Does it violate any EXCLUDE rule?

Only include candidates that satisfy all checks.

DETECTION RULES:
- Scan left-to-right, top-to-bottom.
- Do not stop early.
- Return ONLY "{class_name}" detections.

Return ONLY valid JSON:
{{
  "detections": [
    {{"class": "{class_name}", "label": "", "x_min": <0-100>, "y_min": <0-100>, "x_max": <0-100>, "y_max": <0-100>}}
  ]
}}
Coordinates are percentages of image width/height (0–100 scale).
"""

def build_system_prompt(catalog: str) -> str:
    """Inject object catalog vào system prompt template."""
    return SYSTEM_PROMPT_GEN_TEMPLATE.format(
        catalog=("\n\n" + catalog) if catalog else ""
    )


def build_json_suffix(class_name: str) -> str:
    """Build detection suffix với class name cụ thể."""
    return _JSON_SUFFIX_TEMPLATE.format(class_name=class_name)
