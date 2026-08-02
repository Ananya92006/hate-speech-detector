"""
==========================================================
Test Suite — Comprehensive Testing
==========================================================

PURPOSE:
    Test the entire pipeline with various Hinglish inputs:
    - Slang-based inputs
    - Emoji-heavy inputs
    - Code-mixed text
    - Pure Hindi/English
    
    Run this after training to validate the model works correctly.

TO RUN:
    cd hate-speech-detector
    python tests/test_pipeline.py
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_preprocessing():
    """Test the preprocessing pipeline."""
    from preprocessing.cleaner import TextCleaner
    from preprocessing.emoji_handler import EmojiHandler
    from preprocessing.slang_normalizer import SlangNormalizer
    
    print("\n" + "=" * 70)
    print("  TEST 1: PREPROCESSING PIPELINE")
    print("=" * 70)
    
    cleaner = TextCleaner()
    
    test_cases = [
        # (input, description)
        ("@user123 yeh bakwas hai 😂 #HateContent http://spam.com",
         "Full noise: mentions, emojis, hashtags, URLs"),
        ("TUM PAGAL HO KYA?! 🤡🤡",
         "Uppercase + emojis"),
        ("bohot mast party thi bhai 🎉🥳 #WeekendVibes",
         "Positive Hinglish + emojis"),
        ("saale kameene 😡🔪 nikalo inko",
         "Offensive + violence emojis"),
        ("chapri log hain sab 🤮",
         "Slang-heavy input"),
        ("good morning ☀️ aaj din accha hai ❤️",
         "Mixed language neutral"),
        ("in logon ko desh se nikal do 😡💀",
         "Hate speech + emojis"),
        ("lol bruh that was cringe af 😂",
         "Internet slang heavy"),
    ]
    
    all_passed = True
    for text, description in test_cases:
        cleaned = cleaner.clean(text)
        has_output = len(cleaned) > 0
        status = "✅" if has_output else "❌"
        
        if not has_output:
            all_passed = False
        
        print(f"\n  {status} {description}")
        print(f"     Input:  {text}")
        print(f"     Output: {cleaned}")
    
    # Test step-by-step breakdown
    print(f"\n  📋 Step-by-step for sample text:")
    steps = cleaner.clean_with_steps("@user yeh bakwas hai 😂😡 #hate http://evil.com")
    for step, output in steps.items():
        print(f"     {step}: {output}")
    
    return all_passed


def test_emoji_handler():
    """Test emoji conversion."""
    from preprocessing.emoji_handler import EmojiHandler
    
    print("\n" + "=" * 70)
    print("  TEST 2: EMOJI HANDLER")
    print("=" * 70)
    
    handler = EmojiHandler()
    
    tests = [
        ("😂", "laugh"),
        ("😡", "anger"),
        ("❤️", "love"),
        ("💀", "death"),
        ("🔪", "violence"),
        ("🎉", "celebration"),
    ]
    
    all_passed = True
    for emoji_char, expected in tests:
        result = handler.convert_emojis(emoji_char).strip()
        passed = expected in result.lower()
        status = "✅" if passed else "❌"
        
        if not passed:
            all_passed = False
        
        print(f"  {status} {emoji_char} → {result} (expected: {expected})")
    
    return all_passed


def test_slang_normalizer():
    """Test slang normalization."""
    from preprocessing.slang_normalizer import SlangNormalizer
    
    print("\n" + "=" * 70)
    print("  TEST 3: SLANG NORMALIZER")
    print("=" * 70)
    
    normalizer = SlangNormalizer()
    
    tests = [
        ("bakwas", "nonsense"),
        ("pagal", "crazy"),
        ("bewakoof", "fool"),
        ("zabardast", "amazing"),
        ("mast", "great"),
        ("nalayak", "worthless"),
    ]
    
    all_passed = True
    for slang, expected in tests:
        result = normalizer.normalize(slang)
        passed = expected in result
        status = "✅" if passed else "❌"
        
        if not passed:
            all_passed = False
        
        print(f"  {status} {slang} → {result} (expected: {expected})")
    
    return all_passed


def test_predictor():
    """Test the prediction pipeline."""
    from utils.predictor import HateSpeechPredictor
    
    print("\n" + "=" * 70)
    print("  TEST 4: PREDICTION PIPELINE")
    print("=" * 70)
    
    try:
        predictor = HateSpeechPredictor()
    except Exception as e:
        print(f"  ⚠️  Could not load predictor: {e}")
        return False
    
    # Test texts (from requirements)
    test_texts = [
        # Hinglish sentences (from requirements)
        "yeh bakwas hai 😂",
        "tum pagal ho kya",
        
        # Emoji-heavy inputs
        "party mein bahut maza aaya 🎉🥳🎊",
        "in logon ko maaro 😡🔪💀",
        
        # Slang-based inputs
        "chapri log hain sab yahaan pe",
        "zabardast performance thi",
        
        # Pure neutral
        "aaj mausam bahut accha hai",
        "happy birthday bhai",
        
        # Pure offensive
        "kya ghatiya content hai",
        "bekaar hai tu",
        
        # Pure hate
        "in logon ko desh se nikal do",
        "yeh log insaan nahi hain",
    ]
    
    all_passed = True
    for text in test_texts:
        try:
            result = predictor.predict(text)
            
            # Check that result has all required fields
            required_fields = ['label', 'confidence', 'probabilities', 
                             'cleaned_text', 'emoji']
            has_all_fields = all(f in result for f in required_fields)
            
            # Check confidence is valid
            valid_confidence = 0 <= result['confidence'] <= 1
            
            # Check label is valid  
            valid_label = result['label'] in ['Neutral', 'Offensive', 'Hate Speech']
            
            passed = has_all_fields and valid_confidence and valid_label
            status = "✅" if passed else "❌"
            
            if not passed:
                all_passed = False
            
            print(f"  {status} [{result['emoji']} {result['label']:>12}] "
                  f"({result['confidence']:.1%}) | {text}")
            
        except Exception as e:
            print(f"  ❌ Error on '{text}': {e}")
            all_passed = False
    
    return all_passed


def test_lime_explainer():
    """Test LIME explainability."""
    from utils.predictor import HateSpeechPredictor
    from explainability.lime_explainer import LimeExplainer
    
    print("\n" + "=" * 70)
    print("  TEST 5: LIME EXPLAINABILITY")
    print("=" * 70)
    
    try:
        predictor = HateSpeechPredictor()
        explainer = LimeExplainer(predictor)
    except Exception as e:
        print(f"  ⚠️  Could not load explainer: {e}")
        return False
    
    test_text = "yeh bahut ganda content hai maaro inko"
    
    try:
        # Test basic explanation
        explanation = explainer.explain(
            test_text, num_features=5, num_samples=100
        )
        
        has_words = len(explanation['important_words']) > 0
        has_label = explanation['predicted_label'] in ['Neutral', 'Offensive', 'Hate Speech']
        has_proba = 0 <= explanation['predicted_proba'] <= 1
        
        all_passed = has_words and has_label and has_proba
        
        status = "✅" if all_passed else "❌"
        print(f"  {status} LIME explanation generated successfully")
        print(f"     Prediction: {explanation['predicted_label']} "
              f"({explanation['predicted_proba']:.1%})")
        print(f"     Top words: {explanation['important_words'][:5]}")
        
        # Test highlighted text
        highlighted = explainer.get_highlighted_text(
            test_text, num_features=5, num_samples=100
        )
        
        has_highlight = len(highlighted) > 0
        status = "✅" if has_highlight else "❌"
        print(f"  {status} Highlighted text generated ({len(highlighted)} words)")
        
        # Test formatted text
        formatted = explainer.format_explanation_text(explanation)
        has_format = len(formatted) > 0
        status = "✅" if has_format else "❌"
        print(f"  {status} Formatted explanation text generated")
        
        return all_passed and has_highlight and has_format
        
    except Exception as e:
        print(f"  ❌ LIME test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 70)
    print("  🧪 HATE SPEECH DETECTOR - FULL TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Tests that don't need the model
    results['Preprocessing'] = test_preprocessing()
    results['Emoji Handler'] = test_emoji_handler()
    results['Slang Normalizer'] = test_slang_normalizer()
    
    # Tests that need the model (may fail if not trained yet)
    try:
        results['Predictor'] = test_predictor()
    except Exception as e:
        print(f"\n  ⚠️  Predictor tests skipped: {e}")
        results['Predictor'] = None
    
    try:
        results['LIME Explainer'] = test_lime_explainer()
    except Exception as e:
        print(f"\n  ⚠️  LIME tests skipped: {e}")
        results['LIME Explainer'] = None
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        if passed is True:
            status = "✅ PASSED"
        elif passed is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"  {status:>12}  {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
