# explainer.py
# Generates human-readable maintenance recommendations using multiple LLM providers.
# Fallback chain: Gemini → OpenAI → Anthropic → rule-based mock

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()
from config import (
    LLM_PROVIDER, LLM_MODEL_ANTHROPIC, LLM_MODEL_OPENAI,
    LLM_MAX_TOKENS, SENSOR_NAMES, ACTION_NAMES,
    HEALTH_THRESHOLD_WARN, HEALTH_THRESHOLD_CRIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert aviation maintenance engineer AI assistant.
Your job is to explain maintenance decisions to turbofan engine operators in clear, concise language.
Always:
- Reference the specific sensor readings that drove the decision
- Explain the risk if no action were taken
- Estimate the urgency (immediate / within 24 h / within a week)
- Keep the explanation under 120 words
- Use plain language (avoid heavy jargon)
- End with a one-sentence "Bottom line:" summary"""


def _build_user_prompt(
    sensor_readings: dict[str, float],
    health_score: float,
    action: int,
    step: int,
) -> str:
    health_pct = round(health_score * 100, 1)
    action_name = ACTION_NAMES[action]

    # Classify health
    if health_score > HEALTH_THRESHOLD_WARN:
        health_label = "HEALTHY"
    elif health_score > HEALTH_THRESHOLD_CRIT:
        health_label = "DEGRADED"
    else:
        health_label = "CRITICAL"

    sensor_lines = "\n".join(
        f"  - {name}: {value:.2f}"
        for name, value in sensor_readings.items()
    )

    return f"""Turbofan Engine Health Report — Cycle {step}
====================================
Health Score  : {health_pct}% ({health_label})
Chosen Action : {action_name}

Live Sensor Readings:
{sensor_lines}

Please explain why the AI selected "{action_name}" given these sensor readings and 
health state, and what would happen if this action were ignored.""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Explainer class
# ─────────────────────────────────────────────────────────────────────────────
class MaintenanceExplainer:
    """
    Multi-provider LLM explainer with automatic fallback.

    Tries providers in order: configured provider → other available providers → mock.

    Usage
    -----
    explainer = MaintenanceExplainer()
    text = explainer.explain(sensor_readings, health_score, action, step)
    """

    def __init__(self, provider: str = LLM_PROVIDER):
        self.requested_provider = provider
        self._providers_tried: list[str] = []

    def explain(
        self,
        sensor_readings: dict[str, float],
        health_score: float,
        action: int,
        step: int = 0,
    ) -> str:
        """
        Generate a natural-language explanation for the chosen action.
        Tries multiple LLM providers before falling back to rule-based mock.
        """
        user_prompt = _build_user_prompt(sensor_readings, health_score, action, step)

        # Build priority order: requested provider first, then others
        all_providers = ["gemini", "openai", "anthropic"]
        providers_to_try = [self.requested_provider]
        for p in all_providers:
            if p not in providers_to_try:
                providers_to_try.append(p)

        last_error = None
        for provider in providers_to_try:
            try:
                result = self._try_provider(provider, user_prompt)
                if result and len(result) > 20 and not result.startswith("["):
                    return result
                elif result:
                    last_error = result
            except Exception as e:
                last_error = str(e)
                continue

        # All LLM providers failed — use rule-based mock
        print(f"⚠️  All LLM providers failed (last error: {last_error}). Using rule-based explanation.")
        return self._mock_explain(sensor_readings, health_score, action)

    def _try_provider(self, provider: str, user_prompt: str) -> str | None:
        """Try a specific provider and return the result or None."""
        if provider == "gemini":
            return self._call_gemini(user_prompt)
        elif provider == "openai":
            return self._call_openai(user_prompt)
        elif provider == "anthropic":
            return self._call_anthropic(user_prompt)
        return None

    # ── Gemini ────────────────────────────────────────────────────────────────
    def _call_gemini(self, user_prompt: str) -> str | None:
        import requests
        from config import LLM_MODEL_GEMINI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{LLM_MODEL_GEMINI}:generateContent?key={api_key}"
        )
        data = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": LLM_MAX_TOKENS,
                "temperature": 0.7,
                "thinkingConfig": {
                    "thinkingBudget": 512,
                },
            },
        }
        res = requests.post(url, json=data, timeout=30)

        if res.status_code != 200:
            print(f"⚠️  Gemini API error {res.status_code}: {res.text[:200]}")
            return None

        body = res.json()

        # Check for safety blocks or empty candidates
        if not body.get("candidates"):
            block_reason = body.get("promptFeedback", {}).get("blockReason", "unknown")
            print(f"⚠️  Gemini returned no candidates (blockReason={block_reason})")
            return None

        candidate = body["candidates"][0]

        # Check finish reason
        finish = candidate.get("finishReason", "")
        if finish == "SAFETY":
            print("⚠️  Gemini blocked response for safety reasons")
            return None

        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            return None

        text = parts[0].get("text", "").strip()
        if not text:
            return None

        return text

    # ── OpenAI ────────────────────────────────────────────────────────────────
    def _call_openai(self, user_prompt: str) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            from openai import OpenAI
        except ImportError:
            print("⚠️  openai package not installed.")
            return None

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL_OPENAI,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        return text.strip() if text else None

    # ── Anthropic ─────────────────────────────────────────────────────────────
    def _call_anthropic(self, user_prompt: str) -> str | None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        try:
            import anthropic
        except ImportError:
            print("⚠️  anthropic package not installed.")
            return None

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=LLM_MODEL_ANTHROPIC,
            max_tokens=LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text
        return text.strip() if text else None

    # ── Mock fallback (no API key needed) ─────────────────────────────────────
    @staticmethod
    def _mock_explain(
        sensor_readings: dict[str, float],
        health_score: float,
        action: int,
    ) -> str:
        """Rule-based fallback explanation when no LLM is available."""
        action_name = ACTION_NAMES[action]
        health_pct = round(health_score * 100, 1)

        # Flag high sensors based on C-MAPSS approx limits
        flags = []
        thresholds = {
            "T24": 620,
            "T30": 1595,
            "T50": 1410,
            "Ps30": 47.5,
            "phi": 525,
            "BPR": 8.45,
            "W31": 39,
            "W32": 23.5,
        }
        for sensor, threshold in thresholds.items():
            val = sensor_readings.get(sensor)
            if val is None:
                continue
            if val > threshold:
                flags.append(f"elevated {sensor} ({val:.1f})")

        flag_str = (", ".join(flags) + " — ") if flags else ""

        templates = {
            0: (
                f"Machine health is at {health_pct}%, well within acceptable limits. "
                f"{flag_str}No immediate maintenance action is needed at this time. "
                f"Continue standard monitoring and schedule routine inspection per protocol. "
                f"Bottom line: Machine is operating normally — no action required."
            ),
            1: (
                f"Health has dropped to {health_pct}%. {flag_str}"
                f"An inspection is recommended to identify the root cause before the issue escalates. "
                f"Without inspection, minor issues may worsen into expensive repairs. "
                f"Bottom line: Inspect within 24 hours to prevent further degradation."
            ),
            2: (
                f"Machine health is at {health_pct}% — in the degraded zone. {flag_str}"
                f"Sensor trends indicate mechanical wear that requires targeted repair. "
                f"Delaying repair risks unplanned downtime and cascading component failures. "
                f"Bottom line: Schedule repair within 24 hours to restore safe operation."
            ),
            3: (
                f"Critical health level of {health_pct}% detected. {flag_str}"
                f"Multiple sensor readings show severe degradation beyond repair thresholds. "
                f"Continuing operation risks catastrophic failure and safety hazards. "
                f"Bottom line: Immediate replacement is required — do not continue operation."
            ),
        }
        return templates.get(action, "Unable to generate explanation.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    explainer = MaintenanceExplainer()
    sample_sensors = {
        "T24": 621.5,
        "T30": 1596.2,
        "T50": 1411.8,
        "P30": 554.0,
        "Nf": 2388.1,
        "Nc": 9050.2,
        "Ps30": 47.8,
        "phi": 523.0,
        "NRf": 2388.2,
        "NRc": 8134.0,
        "BPR": 8.48,
        "htBleed": 393.0,
        "W31": 38.5,
        "W32": 23.2,
    }
    print(explainer.explain(sample_sensors, health_score=0.25, action=2, step=150))
