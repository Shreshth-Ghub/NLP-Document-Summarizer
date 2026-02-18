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
            metrics = ['rouge1', 'rouge2', 'rougeL']
        
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    
    def calculate_scores(self, reference, summary):
        """
        Calculate ROUGE scores
        
        Args:
            reference: Reference text (original or gold summary)
            summary: Generated summary to evaluate
        
        Returns:
            dict: ROUGE scores with precision, recall, F1 for each metric
        """
        if not reference or not summary:
            return self._empty_scores()
        
        # Calculate scores
        scores = self.scorer.score(reference, summary)
        
        # Format results
        results = {}
        for metric in self.metrics:
            score = scores[metric]
            results[metric] = {
                'precision': round(score.precision, 4),
                'recall': round(score.recall, 4),
                'fmeasure': round(score.fmeasure, 4)
            }
        
        return results
    
    def _empty_scores(self):
        """Return empty scores structure"""
        results = {}
        for metric in self.metrics:
            results[metric] = {
                'precision': 0.0,
                'recall': 0.0,
                'fmeasure': 0.0
            }
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
        
        # Get F1 scores
        rouge1_f1 = scores['rouge1']['fmeasure']
        rouge2_f1 = scores['rouge2']['fmeasure']
        rougeL_f1 = scores['rougeL']['fmeasure']
        
        # Average F1
        avg_f1 = (rouge1_f1 + rouge2_f1 + rougeL_f1) / 3
        
        # Quality assessment
        if avg_f1 >= 0.50:
            quality = "Excellent"
            quality_color = "success"
        elif avg_f1 >= 0.35:
            quality = "Good"
            quality_color = "info"
        elif avg_f1 >= 0.20:
            quality = "Fair"
            quality_color = "warning"
        else:
            quality = "Poor"
            quality_color = "danger"
        
        return {
            'scores': scores,
            'average_f1': round(avg_f1, 4),
            'quality_level': quality,
            'quality_color': quality_color
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
        
        # Determine winner
        if eval1['average_f1'] > eval2['average_f1']:
            winner = "Summary 1"
        elif eval2['average_f1'] > eval1['average_f1']:
            winner = "Summary 2"
        else:
            winner = "Tie"
        
        return {
            'summary1_evaluation': eval1,
            'summary2_evaluation': eval2,
            'winner': winner
        }


def calculate_rouge(reference, summary, metrics=None):
    """
    Quick function to calculate ROUGE scores
    
    Args:
        reference: Reference text
        summary: Generated summary
        metrics: List of metrics (default: rouge1, rouge2, rougeL)
    
    Returns:
        dict: ROUGE scores
    """
    evaluator = SummaryEvaluator(metrics)
    return evaluator.calculate_scores(reference, summary)


def get_quality_assessment(reference, summary):
    """
    Quick function to get quality assessment
    
    Args:
        reference: Reference text
        summary: Generated summary
    
    Returns:
        dict: Scores + quality level
    """
    evaluator = SummaryEvaluator()
    return evaluator.evaluate_summary_quality(reference, summary)


# Testing
if __name__ == "__main__":
    # Reference text
    reference = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping 
    under a tree in the park. The fox was very agile and jumped high.
    """
    
    # Test summaries
    summary_good = "The quick brown fox jumps over the lazy dog sleeping under a tree."
    summary_fair = "A fox jumped over a dog in the park."
    summary_poor = "There were animals."
    
    print("=== Testing ROUGE Score Evaluation ===\n")
    
    # Initialize evaluator
    evaluator = SummaryEvaluator()
    
    print("Reference text:")
    print(reference)
    print("\n" + "="*60 + "\n")
    
    # Test good summary
    print("Summary 1 (Good):")
    print(summary_good)
    print("\nROUGE Scores:")
    eval1 = evaluator.evaluate_summary_quality(reference, summary_good)
    for metric, values in eval1['scores'].items():
        print(f"{metric.upper()}:")
        print(f"  Precision: {values['precision']:.4f}")
        print(f"  Recall:    {values['recall']:.4f}")
        print(f"  F1:        {values['fmeasure']:.4f}")
    print(f"\nAverage F1: {eval1['average_f1']:.4f}")
    print(f"Quality: {eval1['quality_level']}")
    
    print("\n" + "="*60 + "\n")
    
    # Test fair summary
    print("Summary 2 (Fair):")
    print(summary_fair)
    print("\nROUGE Scores:")
    eval2 = evaluator.evaluate_summary_quality(reference, summary_fair)
    for metric, values in eval2['scores'].items():
        print(f"{metric.upper()}:")
        print(f"  Precision: {values['precision']:.4f}")
        print(f"  Recall:    {values['recall']:.4f}")
        print(f"  F1:        {values['fmeasure']:.4f}")
    print(f"\nAverage F1: {eval2['average_f1']:.4f}")
    print(f"Quality: {eval2['quality_level']}")
    
    print("\n" + "="*60 + "\n")
    
    # Test poor summary
    print("Summary 3 (Poor):")
    print(summary_poor)
    print("\nROUGE Scores:")
    eval3 = evaluator.evaluate_summary_quality(reference, summary_poor)
    for metric, values in eval3['scores'].items():
        print(f"{metric.upper()}:")
        print(f"  Precision: {values['precision']:.4f}")
        print(f"  Recall:    {values['recall']:.4f}")
        print(f"  F1:        {values['fmeasure']:.4f}")
    print(f"\nAverage F1: {eval3['average_f1']:.4f}")
    print(f"Quality: {eval3['quality_level']}")
    
    print("\n" + "="*60 + "\n")
    
    # Compare summaries
    print("=== Comparison ===")
    comparison = evaluator.compare_summaries(reference, summary_good, summary_fair)
    print(f"Summary 1 Average F1: {comparison['summary1_evaluation']['average_f1']:.4f} ({comparison['summary1_evaluation']['quality_level']})")
    print(f"Summary 2 Average F1: {comparison['summary2_evaluation']['average_f1']:.4f} ({comparison['summary2_evaluation']['quality_level']})")
    print(f"Winner: {comparison['winner']}")
