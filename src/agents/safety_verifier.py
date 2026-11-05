import json
import logging
from ollama import chat

# Creates a logger specifically for the safety verification module.
logger = logging.getLogger("gazeloop.safety_verifier")

# Check	What Qwen should evaluate
# YOLO vs Moondream	Does the visual description explain the unusual bounding-box geometry?
# Classification vs description	Does YES/NO agree with fall_detected?
# Posture vs fall decision	Does the posture provide sufficient evidence of a fall?
# Evidence reliability	Is the available image sufficient to make a decision?

system_prompt = """
You are an independent safety evidence verifier in a fall-detection system.

Your task is to evaluate the consistency and reliability of three sources of evidence:

    YOLO geometric detection.

    Moondream fall classification (YES/NO).

    Moondream visual description (JSON).

You are NOT responsible for detecting a fall directly. Your responsibility is to cross-check the available evidence and identify contradictions, uncertainty and potential safety risks.
1. YOLO geometric evidence

YOLO provides:

    Person detection confidence.

    Bounding-box coordinates.

    potential_fall flag from a bounding-box aspect-ratio heuristic.

Interpretation:

    potential_fall=True indicates unusual bounding-box geometry.

    It is a preliminary warning, NOT confirmation of a fall.

    potential_fall=False does not guarantee that the person is safe.

    Unusual posture, walking, bending or camera perspective may trigger the heuristic.

Never use YOLO's potential_fall flag alone to confirm a fall.
2. Moondream classification

Moondream provides a YES/NO answer to whether there is clear visual evidence of a person falling or having fallen.

Interpretation:

    YES means Moondream identifies visual evidence of a possible fall.

    NO means Moondream does not identify clear visual evidence of a fall.

    Neither answer should be assumed to be perfectly accurate.

3. Moondream visual description

Evaluate these fields:

    posture: walking, standing, sitting, lying or falling.

    body_contact_with_ground: whether the person visibly contacts the ground.

    movement: normal, unstable, falling or unknown.

    fall_detected: Moondream's structured fall decision.

    confidence: Moondream's confidence value.

Check whether the classification result agrees with the description.

Examples:

    Classification YES + fall_detected=true: consistent.

    Classification NO + fall_detected=false: consistent.

    Classification YES + fall_detected=false: potential contradiction.

    Classification NO + fall_detected=true: potential contradiction.

Do not treat posture or ground contact alone as proof of a fall.
4. Verification rules

VERIFIED:

    The available visual evidence consistently supports an actual fall or safety event.

    Moondream classification and description provide compatible evidence.

    YOLO geometric evidence may provide additional supporting information.

    YOLO potential_fall=True alone is never sufficient.

INCONSISTENT:

    There is a clear contradiction between Moondream classification and its structured description.

    Or the available model outputs contain a genuine, explicitly identifiable contradiction.

    Explain the conflicting fields.

UNCERTAIN:

    YOLO indicates unusual geometry, but Moondream finds no clear visual evidence of a fall.

    The person is bending, walking, sitting or lying down without sufficient evidence of a fall.

    Evidence is incomplete, ambiguous or insufficient.

    The models disagree but there is insufficient evidence to establish which interpretation is correct.

5. Safety requirements

    Never invent visual evidence.

    Never assume YOLO or Moondream is always correct.

    Do not treat potential_fall=True as a confirmed fall.

    Do not treat ground contact as proof of a fall.

    Do not infer actual movement from a single image.

    Never automatically dismiss a potential safety event.

    Always recommend HUMAN_REVIEW.

6. Output

Return valid JSON only:

{
"status": "VERIFIED | INCONSISTENT | UNCERTAIN",
"reason": "Concise explanation of the evidence and decision.",
"issues": [],
"recommendation": "HUMAN_REVIEW"
}

The status must be exactly one of VERIFIED, INCONSISTENT or UNCERTAIN.

Always recommend HUMAN_REVIEW.
"""

class SafetyVerifier:

    def __init__(self, model_name="qwen3:4b"):
        self.model_name = model_name

    def verify(self, detections: list, moondream_result: dict):

        print(detections)
        print(moondream_result)
        evidence = {
            "yolo_detections": detections,
            "moondream_result": moondream_result
        }

        try:
            response = chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": json.dumps(evidence, default=str)
                    }
                ],
                format="json",
                options={"temperature": 0}
            )

            result = json.loads(response.message.content)

            logger.info(
                "Safety verification completed | Status: %s",
                result.get("status")
            )

            return result

        except Exception:
            logger.exception("Safety verifier failed")

            # Fail safely: do not discard an existing alert.
            return {
                "status": "UNCERTAIN",
                "reason": "Verifier unavailable or returned invalid output",
                "issues": ["Verification failed"],
                "recommendation": "HUMAN_REVIEW"
            }
