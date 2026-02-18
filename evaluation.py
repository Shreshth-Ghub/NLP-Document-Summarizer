"""
evaluation.py
ROUGE Score Evaluation for Summary Quality

Calculates ROUGE metrics to evaluate summary quality.
"""

from rouge_score import rouge_scorer


class SummaryEvaluator:
    """Calculate ROUGE scores for summary evaluation"""

    def __init__(self, metrics=None):
        """
        Initialize ROUGE scorer

        Args:
            metrics: List of ROUGE metrics to calculate
                     Default: ['rouge1', 'rouge2', 'rougeL']
        """
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]

        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 from precision and recall."""
        if precision is None or recall is None:
            return 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def calculate_scores(self, reference, summary):
        """
        Calculate ROUGE scores

        Args:
            reference: Reference text (original or gold summary)
            summary: Generated summary to evaluate

        Returns:
            dict: ROUGE scores with precision, recall, f1 for each metric,
                  using keys 'rouge-1', 'rouge-2', 'rouge-l'
        """
        if not reference or not summary:
            return self._empty_scores()

        # Calculate scores from library
        raw_scores = self.scorer.score(reference, summary)

        # Map library metric names -> template-friendly keys
        metric_map = {
            "rouge1": "rouge-1",
            "rouge2": "rouge-2",
            "rougeL": "rouge-l",
        }

        results = {}
        for metric in self.metrics:
            score = raw_scores[metric]
            p = float(score.precision)
            r = float(score.recall)
            f1 = self._compute_f1(p, r)

            key = metric_map.get(metric, metric)
            results[key] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }

        return results

    def _empty_scores(self):
        """Return empty scores structure"""
        keys = ["rouge-1", "rouge-2", "rouge-l"]
        results = {}
        for k in keys:
            results[k] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return results

    def evaluate_summary_quality(self, reference, summary):
        """
        Evaluate summary and provide quality assessment

        Args:
            reference: Reference text
            summary: Generated summary

        Returns:
            dict: Scores + quality level assessment
        """
        scores = self.calculate_scores(reference, summary)

        # Get F1 scores (template expects rouge-1/2/l)
        rouge1_f1 = scores["rouge-1"]["f1"]
        rouge2_f1 = scores["rouge-2"]["f1"]
        rougeL_f1 = scores["rouge-l"]["f1"]

        # Average F1
        avg_f1 = (rouge1_f1 + rouge2_f1 + rougeL_f1) / 3.0

        # Quality assessment
        if avg_f1 >= 0.50:
            quality = "excellent"
            quality_color = "success"
        elif avg_f1 >= 0.35:
            quality = "good"
            quality_color = "info"
        elif avg_f1 >= 0.20:
            quality = "fair"
            quality_color = "warning"
        else:
            quality = "poor"
            quality_color = "danger"

        return {
            "scores": scores,
            "average_f1": round(avg_f1, 4),
            "quality_level": quality,
            "quality_color": quality_color,
        }

    def compare_summaries(self, reference, summary1, summary2):
        """
        Compare two summaries against reference

        Args:
            reference: Reference text
            summary1: First summary
            summary2: Second summary

        Returns:
            dict: Comparison results
        """
        eval1 = self.evaluate_summary_quality(reference, summary1)
        eval2 = self.evaluate_summary_quality(reference, summary2)

        if eval1["average_f1"] > eval2["average_f1"]:
            winner = "summary1"
        elif eval2["average_f1"] > eval1["average_f1"]:
            winner = "summary2"
        else:
            winner = "tie"

        return {
            "summary1_evaluation": eval1,
            "summary2_evaluation": eval2,
            "winner": winner,
        }


def calculate_rouge(reference, summary, metrics=None):
    """
    Quick function to calculate ROUGE scores
    """
    evaluator = SummaryEvaluator(metrics)
    return evaluator.calculate_scores(reference, summary)


def get_quality_assessment(reference, summary):
    """
    Quick function to get quality assessment
    """
    evaluator = SummaryEvaluator()
    return evaluator.evaluate_summary_quality(reference, summary)


# Testing
if __name__ == "__main__":
    reference = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping 
    under a tree in the park. The fox was very agile and jumped high.
    """

    summary_good = "The quick brown fox jumps over the lazy dog sleeping under a tree."
    summary_fair = "A fox jumped over a dog in the park."
    summary_poor = "There were animals."

    print("=== Testing ROUGE Score Evaluation ===\n")

    evaluator = SummaryEvaluator()

    for label, s in [("Good", summary_good), ("Fair", summary_fair), ("Poor", summary_poor)]:
        print(f"Summary ({label}): {s}")
        eval_res = evaluator.evaluate_summary_quality(reference, s)
        print("Scores:", eval_res["scores"])
        print("Average F1:", eval_res["average_f1"])
        print("Quality:", eval_res["quality_level"])
        print("=" * 60)
