import argparse
import asyncio
import logging
import sys

# Ensure NLTK sentence tokenizer is present (Leonardo needs it)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "donatello"))
sys.path.append(str(ROOT_DIR / "leonardo"))

from donatello.drafter import DrugInfoAgent  # type: ignore
from leonardo import fact_selfcheck_pipeline  # type: ignore

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PASSAGE_THRESHOLD: float = 0.35   # acceptable Leonardo passage_score (0 = perfect)
MAX_ITERATIONS: int = 2           # max Donatello → Leonardo rounds
N_SAMPLES_FOR_LEONARDO: int = 5   # stochastic samples per Leonardo run

# ---------------------------------------------------------------------------
async def _single_round(
    agent: DrugInfoAgent,
    question: str,
    previous_answer: str | None,
    logger: logging.Logger,
) -> tuple[str, float, dict]:
    """Run Donatello once then let Leonardo critique it."""

    if previous_answer is None:
        logger.info("Donatello drafting initial answer …")
        answer = await agent.answer(question)
    else:
        logger.info("Donatello revising with Leonardo feedback …")
        answer = await agent.revise(question, prev_answer=previous_answer, external_feedback=_cached_feedback)

    logger.info("Donatello produced %d characters.", len(answer))

    # Leonardo evaluation
    logger.info("Leonardo evaluating factual consistency …")
    review = await fact_selfcheck_pipeline(
            prompt = question,
            response_text = answer,
            n_samples = N_SAMPLES_FOR_LEONARDO,
            method = "kg",
            agg_method = "mean",  # <<<
    )
    passage_score = float(review["passage_score"])
    logger.info("Leonardo passage_score = %.3f", passage_score)

    return answer, passage_score, review


async def run(question: str, data_dir: Path, cache_dir: Path, logger: logging.Logger) -> str:
    """Orchestrate up to MAX_ITERATIONS correction rounds."""
    logger.info("🏗  Initialising Donatello …")
    agent = DrugInfoAgent(data_dir=data_dir, cache_dir=cache_dir)

    answer: str | None = None
    passage_score: float = 1.0
    leonardo_raw: dict = {}
    global _cached_feedback
    _cached_feedback = ""

    for itr in range(MAX_ITERATIONS + 1):
        logger.info("================ ROUND %d ================", itr + 1)
        answer, passage_score, leonardo_raw = await _single_round(agent, question, answer, logger)

        if passage_score <= PASSAGE_THRESHOLD:
            logger.info("✅ Accepted by Leonardo (score %.3f).", passage_score)
            break

        if itr >= MAX_ITERATIONS:
            logger.warning("Reached maximum iterations – accepting last answer.")
            break

        # build feedback from worst facts
        # build feedback from worst facts
        bad_facts = [
            (fact, sc) for fact, sc in leonardo_raw["fact_scores"].items() if sc >= 0.8
        ]
        bad_facts.sort(key=lambda x: x[1], reverse=True)

        feedback_lines = []
        for fact, sc in bad_facts[:15]:
            if isinstance(fact, tuple) and len(fact) == 3:
                h, r, t = fact
                feedback_lines.append(f"{h} {r} {t} – score {sc:.2f}")
            else:
                feedback_lines.append(f"{str(fact)} – score {sc:.2f}")

        _cached_feedback = "\n".join(feedback_lines) or "General factual inconsistencies detected."
        logger.debug("Feedback for Donatello:\n%s", _cached_feedback)

    logger.info("Final passage_score = %.3f", passage_score)
    return answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global PASSAGE_THRESHOLD, MAX_ITERATIONS  # declare first!

    p = argparse.ArgumentParser(
        description="Donatello ↔ Leonardo loop controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("question", nargs="+", help="User question")
    p.add_argument("--data-dir", default="structured_drug_data", help="SPL XML directory")
    p.add_argument("--cache-dir", default=".", help="Vector cache directory")
    p.add_argument("--threshold", type=float, default=PASSAGE_THRESHOLD, help="Leonardo acceptance threshold")
    p.add_argument("--max-iter", type=int, default=MAX_ITERATIONS, help="Maximum correction rounds")
    args = p.parse_args()

    PASSAGE_THRESHOLD = max(0.0, min(1.0, args.threshold))
    MAX_ITERATIONS = max(0, args.max_iter)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("Michelangelo")

    question_str = " ".join(args.question)

    final_answer = asyncio.run(run(question_str, Path(args.data_dir), Path(args.cache_dir), logger))

    # per requirements – print just the vetted answer
    print("\n" + final_answer)


if __name__ == "__main__":
    main()
