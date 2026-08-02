"""
==========================================================
LIME Explainability Module
==========================================================

PURPOSE:
    Make the model's predictions INTERPRETABLE. When the model 
    says "This is Hate Speech", LIME tells us WHY — by 
    highlighting which words drove the decision.

WHAT IS LIME?
    LIME = Local Interpretable Model-Agnostic Explanations
    
    Simple explanation:
    1. Take an input text: "in logon ko maaro saale"
    2. Create many "perturbed" versions by randomly removing words:
       - "in logon ko ___ saale"
       - "in ___ ko maaro ___"
       - "___ logon ___ maaro saale"
    3. Feed ALL perturbed versions through the model
    4. See which words, when removed, change the prediction most
    5. Words that change prediction the most = most important!
    
    Think of it like this:
    If removing "maaro" makes the model change from "Hate Speech" 
    to "Neutral", then "maaro" was a KEY word for the prediction.

WHY IS THIS IMPORTANT?
    - Makes AI decisions transparent (required in many regulations)
    - Helps debug model errors
    - Builds trust with users
    - Essential for academic presentations!
"""

import os
import sys
import numpy as np
from lime.lime_text import LimeTextExplainer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ID_TO_LABEL, LIME_NUM_FEATURES, LIME_NUM_SAMPLES


class LimeExplainer:
    """
    LIME-based explainability for hate speech predictions.
    
    This class wraps the LimeTextExplainer to provide:
    1. Word-level importance explanations
    2. HTML visualizations
    3. Structured explanation data for the UI
    
    Usage:
        from utils.predictor import HateSpeechPredictor
        
        predictor = HateSpeechPredictor()
        explainer = LimeExplainer(predictor)
        
        explanation = explainer.explain("in logon ko maaro saale")
        print(explanation['important_words'])
    """
    
    def __init__(self, predictor):
        """
        Initialize the LIME explainer.
        
        Args:
            predictor: HateSpeechPredictor instance
                      Must have a predict_proba(texts) method
                      that returns numpy array of probabilities
        """
        self.predictor = predictor
        
        # Get class names from config
        self.class_names = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL.keys())]
        
        # Create the LIME text explainer
        # split_expression: How to split text into words
        # We use a simple regex that splits on whitespace
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r'\s+',   # Split on whitespace
            random_state=42            # For reproducibility
        )
        
        print(f"  [OK] LIME Explainer initialized")
        print(f"     Classes: {self.class_names}")
    
    def explain(self, text, num_features=None, num_samples=None):
        """
        Generate a LIME explanation for the given text.
        
        This is the main method you should use. It returns
        a structured explanation with word importances.
        
        Args:
            text (str): The text to explain (should be preprocessed)
            num_features (int): Number of top features/words to show
            num_samples (int): Number of perturbations to generate
                             More = accurate but slower
        
        Returns:
            dict: {
                'text': str,                    # Input text
                'predicted_label': str,          # What the model predicted
                'predicted_proba': float,        # Confidence
                'important_words': list[tuple],  # [(word, weight), ...]
                'explanation_object': obj,       # Raw LIME explanation
                'word_importances': dict         # {word: weight, ...}
            }
        """
        if num_features is None:
            num_features = LIME_NUM_FEATURES
        if num_samples is None:
            num_samples = LIME_NUM_SAMPLES
        
        # Get the base prediction first
        base_probs = self.predictor.predict_proba([text])[0]
        predicted_class = int(np.argmax(base_probs))
        predicted_label = ID_TO_LABEL[predicted_class]
        predicted_proba = float(base_probs[predicted_class])
        
        # Generate LIME explanation
        # This calls predict_proba many times with perturbed text
        explanation = self.explainer.explain_instance(
            text,
            self.predictor.predict_proba,  # The prediction function
            num_features=num_features,
            num_samples=num_samples,
            labels=[predicted_class]  # Explain the predicted class
        )
        
        # Extract word importances
        # as_list() returns [(feature, weight), ...]
        # Positive weight = supports the prediction
        # Negative weight = opposes the prediction
        word_weights = explanation.as_list(label=predicted_class)
        
        # Create a clean word importance dict
        word_importances = {}
        for word, weight in word_weights:
            word_importances[word] = weight
        
        return {
            'text': text,
            'predicted_label': predicted_label,
            'predicted_proba': predicted_proba,
            'all_probabilities': {
                ID_TO_LABEL[i]: float(base_probs[i]) 
                for i in range(len(base_probs))
            },
            'important_words': word_weights,
            'explanation_object': explanation,
            'word_importances': word_importances
        }
    
    def get_html_explanation(self, text, num_features=None, num_samples=None):
        """
        Generate an HTML visualization of the explanation.
        
        This creates a colored HTML where:
        - Words supporting the prediction are highlighted in one color
        - Words opposing the prediction are in another color
        
        Useful for embedding in web apps!
        
        Args:
            text (str): Text to explain
            num_features (int): Number of features to show
            num_samples (int): Number of perturbations
            
        Returns:
            str: HTML string with the explanation
        """
        explanation_data = self.explain(text, num_features, num_samples)
        explanation = explanation_data['explanation_object']
        
        # Get HTML from LIME
        html = explanation.as_html()
        
        return html
    
    def get_highlighted_text(self, text, num_features=None, num_samples=None):
        """
        Generate highlighted text data for custom UI rendering.
        
        Returns word-level data that can be used to create
        custom visualizations in Streamlit or any frontend.
        
        Args:
            text (str): Text to explain
            
        Returns:
            list[dict]: List of word data with importance info
            
            Each dict: {
                'word': str,
                'importance': float,
                'is_important': bool,
                'supports_prediction': bool,
                'color': str (CSS color)
            }
        """
        explanation_data = self.explain(text, num_features, num_samples)
        word_importances = explanation_data['word_importances']
        
        words = text.split()
        highlighted = []
        
        for word in words:
            importance = word_importances.get(word, 0.0)
            
            # Determine color based on importance
            if importance > 0:
                # Supports prediction — red intensity based on weight
                intensity = min(255, int(abs(importance) * 500))
                color = f"rgba(231, 76, 60, {min(1.0, abs(importance) * 3):.2f})"
                supports = True
            elif importance < 0:
                # Opposes prediction — green intensity
                intensity = min(255, int(abs(importance) * 500))
                color = f"rgba(46, 204, 113, {min(1.0, abs(importance) * 3):.2f})"
                supports = False
            else:
                color = "transparent"
                supports = None
            
            highlighted.append({
                'word': word,
                'importance': float(importance),
                'is_important': abs(importance) > 0.01,
                'supports_prediction': supports,
                'color': color
            })
        
        return highlighted
    
    def format_explanation_text(self, explanation_data):
        """
        Format explanation as human-readable text.
        
        Args:
            explanation_data (dict): Output from explain()
            
        Returns:
            str: Formatted explanation string
        """
        lines = []
        lines.append(f"Text: \"{explanation_data['text']}\"")
        lines.append(f"Prediction: {explanation_data['predicted_label']} "
                     f"({explanation_data['predicted_proba']:.1%})")
        lines.append("")
        lines.append("Important words (sorted by influence):")
        
        # Sort by absolute importance
        sorted_words = sorted(
            explanation_data['important_words'],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for word, weight in sorted_words:
            direction = "(+) supports" if weight > 0 else "(-) opposes"
            lines.append(f"  {word:>15}: {weight:+.4f} ({direction})")
        
        return "\n".join(lines)


# ============================================
# DEMONSTRATION
# ============================================
if __name__ == "__main__":
    from utils.predictor import HateSpeechPredictor
    
    print("=" * 60)
    print("  LIME EXPLAINABILITY - DEMO")
    print("=" * 60)
    
    # Initialize
    predictor = HateSpeechPredictor()
    explainer = LimeExplainer(predictor)
    
    # Test texts
    test_texts = [
        "yeh bahut ganda content hai log maaro",
        "aaj mausam bahut accha hai enjoy karo",
        "bakwas hai yeh totally useless",
    ]
    
    for text in test_texts:
        print(f"\n{'─' * 60}")
        explanation = explainer.explain(text, num_features=5, num_samples=100)
        print(explainer.format_explanation_text(explanation))
