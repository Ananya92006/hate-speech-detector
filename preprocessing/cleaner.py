"""
==========================================================
Text Cleaner Module - Complete Preprocessing Pipeline
==========================================================

PURPOSE:
    This is the MAIN preprocessing module that combines ALL
    cleaning steps into a single, easy-to-use pipeline.

PIPELINE ORDER:
    1. Lowercasing
    2. URL removal
    3. Mention (@user) removal
    4. Hashtag cleaning (keep text, remove #)
    5. Emoji conversion (emoji → text)
    6. Slang normalization (slang → standard English)
    7. Special character removal
    8. Extra whitespace cleanup

WHY THIS ORDER MATTERS:
    - Emojis must be converted BEFORE special char removal
      (otherwise they'd be deleted!)
    - Lowercasing should happen BEFORE slang normalization
      (slang dict uses lowercase keys)
    - URLs/mentions should be removed first as they add noise
"""

import re
import nltk
from preprocessing.emoji_handler import EmojiHandler
from preprocessing.slang_normalizer import SlangNormalizer

# Download required NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextCleaner:
    """
    Complete text preprocessing pipeline for hate speech detection.
    
    This class orchestrates all cleaning steps in the correct order.
    Each step is also available individually for debugging.
    
    Usage:
        cleaner = TextCleaner()
        clean_text = cleaner.clean("yeh bakwas hai 😂 @user #hatecontent")
        # Output: "yeh nonsense hai laugh"
    """
    
    def __init__(self, remove_stopwords=False):
        """
        Initialize all preprocessing components.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords.
                                     Default False because for hate speech,
                                     even common words provide context.
        """
        self.emoji_handler = EmojiHandler()
        self.slang_normalizer = SlangNormalizer()
        self.remove_stopwords_flag = remove_stopwords
        
        # Load English stopwords if needed
        if remove_stopwords:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()
        
        # Compile regex patterns once for efficiency
        # Each pattern handles a specific type of noise
        
        # Matches URLs: http://..., https://..., www.....
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            r'[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}'
        )
        
        # Matches @mentions: @username123
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        
        # Matches #hashtags: #HateSpeech → HateSpeech
        self.hashtag_pattern = re.compile(r'#(\w+)')
        
        # Matches special characters (keeps letters, numbers, spaces)
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        # Matches multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
    
    def to_lowercase(self, text):
        """
        Step 1: Convert text to lowercase.
        
        WHY: Makes matching consistent. "HATE" and "hate" should 
        be treated the same way.
        """
        return text.lower()
    
    def remove_urls(self, text):
        """
        Step 2: Remove URLs from text.
        
        WHY: URLs are noise — they don't help classify hate speech.
        Example: "check this http://evil.com bakwas" → "check this  bakwas"
        """
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text):
        """
        Step 3: Remove @mentions from text.
        
        WHY: Usernames are not relevant to the content's hatefulness.
        Example: "@user123 tu pagal hai" → " tu pagal hai"
        """
        return self.mention_pattern.sub('', text)
    
    def clean_hashtags(self, text):
        """
        Step 4: Remove # symbol but keep the text.
        
        WHY: Hashtag TEXT can be meaningful (e.g., #GoBackMuslims),
        but the # symbol itself is noise.
        Example: "#HateContent" → "HateContent"
        """
        return self.hashtag_pattern.sub(r'\1', text)
    
    def convert_emojis(self, text):
        """
        Step 5: Convert emojis to text descriptions.
        
        WHY: Emojis carry sentiment (😂=mockery, 😡=anger).
        This step captures that information as text.
        Example: "bakwas hai 😂" → "bakwas hai laugh"
        """
        return self.emoji_handler.convert_emojis(text)
    
    def normalize_slang(self, text):
        """
        Step 6: Replace slang with standard English.
        
        WHY: "bakwas" → "nonsense" helps the model understand intent.
        Example: "pagal hai tu" → "crazy hai tu"
        """
        return self.slang_normalizer.normalize(text)
    
    def remove_special_chars(self, text):
        """
        Step 7: Remove special characters.
        
        WHY: Punctuation and symbols add noise without helping
        hate speech classification.
        Example: "hello!! @#$ world" → "hello world"
        """
        return self.special_char_pattern.sub(' ', text)
    
    def remove_extra_whitespace(self, text):
        """
        Step 8: Remove extra whitespace and trim.
        
        WHY: Previous steps may introduce extra spaces.
        Example: "hello   world  " → "hello world"
        """
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def remove_stopwords_fn(self, text):
        """
        Optional: Remove common English stopwords.
        
        NOTE: This is OFF by default because stopwords can
        provide important context in hate speech detection.
        "you are trash" vs "trash" — "you are" matters!
        """
        if not self.remove_stopwords_flag:
            return text
        words = text.split()
        return ' '.join(w for w in words if w not in self.stopwords)
    
    def clean(self, text):
        """
        Run the COMPLETE preprocessing pipeline.
        
        This is the main method you should use. It applies
        all cleaning steps in the correct order.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Fully cleaned and preprocessed text
            
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.clean("@user yeh bakwas hai 😂 #hatecontent http://spam.com")
            'yeh nonsense is laugh hatecontent'
        """
        if not isinstance(text, str):
            return ""
        
        # Apply each step in order
        text = self.to_lowercase(text)          # Step 1
        text = self.remove_urls(text)           # Step 2
        text = self.remove_mentions(text)       # Step 3
        text = self.clean_hashtags(text)        # Step 4
        text = self.convert_emojis(text)        # Step 5
        text = self.normalize_slang(text)       # Step 6
        text = self.remove_special_chars(text)  # Step 7
        text = self.remove_extra_whitespace(text)  # Step 8
        text = self.remove_stopwords_fn(text)   # Optional
        
        return text
    
    def clean_with_steps(self, text):
        """
        Run the pipeline and show the output of EACH step.
        
        Useful for debugging and demonstrating the pipeline
        in presentations.
        
        Args:
            text (str): Raw input text
            
        Returns:
            dict: Dictionary mapping step names to their outputs
        """
        steps = {}
        
        steps["0_original"] = text
        
        text = self.to_lowercase(text)
        steps["1_lowercase"] = text
        
        text = self.remove_urls(text)
        steps["2_no_urls"] = text
        
        text = self.remove_mentions(text)
        steps["3_no_mentions"] = text
        
        text = self.clean_hashtags(text)
        steps["4_clean_hashtags"] = text
        
        text = self.convert_emojis(text)
        steps["5_emojis_converted"] = text
        
        text = self.normalize_slang(text)
        steps["6_slang_normalized"] = text
        
        text = self.remove_special_chars(text)
        steps["7_no_special_chars"] = text
        
        text = self.remove_extra_whitespace(text)
        steps["8_final"] = text
        
        return steps


# ============================================
# DEMONSTRATION - Before vs After
# ============================================
if __name__ == "__main__":
    cleaner = TextCleaner()
    
    # Test cases showing real-world social media text
    test_texts = [
        "@hater123 yeh bakwas hai 😂😡 #HateSpeech http://spam.com",
        "TUM PAGAL HO KYA?! 🤡🤡 @user555",
        "bohot mast party thi bhai 🎉🥳 #WeekendVibes",
        "saale kameene 😡🔪 inko maaro #violence https://evil.link",
        "chapri log hain sab 🤮 #cringe @everyone",
        "good morning!! ☀️ aaj din accha hai ❤️",
        "in logon ko desh se nikal do 😡💀 @admin #GoBack",
    ]
    
    print("=" * 70)
    print("  TEXT CLEANER - COMPLETE PIPELINE DEMO")
    print("=" * 70)
    
    for text in test_texts:
        print(f"\n  {'BEFORE':>8}: {text}")
        print(f"  {'AFTER':>8}: {cleaner.clean(text)}")
        print("-" * 70)
    
    # Show detailed step-by-step for one example
    print("\n" + "=" * 70)
    print("  STEP-BY-STEP BREAKDOWN")
    print("=" * 70)
    
    example = "@hater123 yeh bakwas hai 😂😡 #HateSpeech http://spam.com"
    steps = cleaner.clean_with_steps(example)
    
    for step_name, step_output in steps.items():
        print(f"  {step_name:>25}: {step_output}")
