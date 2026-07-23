"""
explain_service.py
====================
Turns a numeric run result (from backend/service.py's run_technique) into
a plain-English explanation. Uses the LLM if configured; otherwise falls
back to genuinely-informative templated logic grounded in this project's
own measured findings (see REPORT.md) rather than a generic "it depends."
"""

from backend import llm_client

SYSTEM_PROMPT = """You explain quantum error mitigation experiment results in plain English \
to someone who is curious but not a quantum physicist. You will be given the numeric \
result of one run (technique, noise strength, ideal/raw/mitigated values, error reduction \
percentage) plus a short technical description of the technique. Write 2-4 short sentences: \
say plainly whether mitigation helped, by how much, and give one honest, concrete reason tied \
to the technique's known behavior for why it did or didn't. Avoid jargon where you can, and \
don't pad with generic disclaimers."""


def _severity_word(pct: float) -> str:
    if pct > 50:
        return "substantially"
    if pct > 15:
        return "noticeably"
    if pct > 5:
        return "modestly"
    if pct > -5:
        return "negligibly"
    if pct > -50:
        return "noticeably worse"
    return "substantially worse"


_TECHNIQUE_CAVEATS = {
    "zne": (
        "Zero-Noise Extrapolation fits a curve through a few noise-amplified data points; "
        "on short/shallow circuits there are too few, too-similar points for that fit to be "
        "reliable, which can make it undershoot or overshoot the true value."
    ),
    "mem": (
        "Measurement Error Mitigation only targets classical readout error, which is easy to "
        "characterize precisely -- that's why it tends to be the most consistently reliable "
        "technique here."
    ),
    "pec": (
        "Probabilistic Error Cancellation's accuracy depends heavily on how closely its assumed "
        "noise strength matches the real noise, and its variance depends on how many samples "
        "were drawn -- both were deliberately kept modest here for speed."
    ),
    "cdr": (
        "Clifford Data Regression learns an empirical correction from training circuits rather "
        "than assuming a fixed noise model, which tends to make it degrade more gracefully as "
        "noise increases."
    ),
    "ddd": (
        "Dynamical Decoupling only helps during idle windows in a circuit -- on tightly "
        "scheduled small circuits there's little idle time for it to act on."
    ),
    "vd": (
        "Virtual Distillation's own ancilla and controlled-SWAP gates add noise of their own, "
        "which can partly offset the exponential error suppression it's designed to provide, "
        "especially on shallow, low-qubit circuits."
    ),
}


def _template_explanation(result: dict) -> str:
    pct = result["error_reduction_pct"]
    technique = result["technique"]
    technique_name = result["technique_name"]
    severity = _severity_word(pct)
    caveat = _TECHNIQUE_CAVEATS.get(technique, "")

    if pct > 5:
        verdict = (
            f"{technique_name} {severity} improved this result: the error shrank from "
            f"{result['raw_error']:.4f} to {result['mitigated_error']:.4f} "
            f"({pct:.1f}% error reduction) at noise strength {result['noise_strength']}."
        )
    elif pct < -5:
        verdict = (
            f"{technique_name} actually made this result {severity} here: the error grew from "
            f"{result['raw_error']:.4f} to {result['mitigated_error']:.4f} "
            f"({abs(pct):.1f}% worse) at noise strength {result['noise_strength']}."
        )
    else:
        verdict = (
            f"{technique_name} left this result {severity} changed: error went from "
            f"{result['raw_error']:.4f} to {result['mitigated_error']:.4f} "
            f"at noise strength {result['noise_strength']}."
        )

    return f"{verdict} {caveat}".strip()


def explain_result(result: dict) -> dict:
    llm_answer = None
    if llm_client.is_available():
        user_message = (
            f"Technique: {result['technique_name']} ({result['technique']})\n"
            f"Benchmark circuit: {result['benchmark']}\n"
            f"Observable: {result['observable']}\n"
            f"Noise strength: {result['noise_strength']}\n"
            f"Ideal value: {result['ideal']:.4f}\n"
            f"Raw (noisy) value: {result['raw']:.4f} (error {result['raw_error']:.4f})\n"
            f"Mitigated value: {result['mitigated']:.4f} (error {result['mitigated_error']:.4f})\n"
            f"Error reduction: {result['error_reduction_pct']:.1f}%\n"
            f"Known technique behavior: {_TECHNIQUE_CAVEATS.get(result['technique'], 'n/a')}"
        )
        llm_answer = llm_client.generate(SYSTEM_PROMPT, user_message, max_tokens=300)

    used_llm = llm_answer is not None
    explanation = llm_answer if used_llm else _template_explanation(result)

    return {"explanation": explanation, "used_llm": used_llm}
