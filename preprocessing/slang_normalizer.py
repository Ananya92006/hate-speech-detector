"""
==========================================================
Slang Normalizer Module
==========================================================

PURPOSE:
    Hinglish social media text is filled with slang, 
    abbreviations, and informal words that standard NLP
    models struggle with. This module normalizes them
    to standard English equivalents.

HOW IT WORKS:
    Maintains a dictionary of common Hinglish slang words
    and their English translations. During preprocessing,
    each word is checked against this dictionary and 
    replaced if a match is found.

WHY IS THIS IMPORTANT?
    Without normalization:
    - "bakwas" would be treated as an unknown word
    - "pagal" wouldn't be recognized as "crazy"
    - The model would miss important hate/offensive signals

EXAMPLE:
    Input:  "yeh bakwas hai, tu pagal hai kya"
    Output: "yeh nonsense hai, tu crazy hai kya"
"""

import re


class SlangNormalizer:
    """
    Normalizes Hinglish slang terms to standard English equivalents.
    
    The dictionary covers:
    1. Common Hinglish abusive terms
    2. Internet slang & abbreviations
    3. Informal Hindi words commonly used in code-mixed text
    4. Social media specific shortforms
    """
    
    def __init__(self):
        """
        Initialize the slang dictionary.
        
        The dictionary is organized by category for easy 
        maintenance and extension.
        """
        # ============================================
        # HINGLISH SLANG DICTIONARY
        # ============================================
        # Each entry: "slang_word": "standard_english_equivalent"
        
        self.slang_dict = {
            # ----- Abusive / Offensive Terms -----
            # These are common in hate speech and need normalization
            "bakwas": "nonsense",
            "bakwaas": "nonsense",
            "pagal": "crazy",
            "paagal": "crazy",
            "bewakoof": "fool",
            "bewkoof": "fool",
            "bevkoof": "fool",
            "gadha": "donkey",
            "gadhe": "donkey",
            "ullu": "owl fool",
            "buddhu": "stupid",
            "nalayak": "worthless",
            "nikamma": "useless",
            "nikammi": "useless",
            "kameena": "mean person",
            "kameeni": "mean person",
            "kamina": "mean person",
            "haramkhor": "scoundrel",
            "harami": "illegitimate",
            "lafanga": "rogue",
            "tharki": "pervert",
            "chapri": "lowlife",
            "chapris": "lowlifes",
            "chirkut": "worthless person",
            "chutiya": "idiot",
            "chu": "idiot",
            "gandu": "idiot",
            "mc": "abusive word",
            "bc": "abusive word",
            "saala": "abusive term",
            "saale": "abusive term",
            "saaley": "abusive term",
            "kutte": "dog insult",
            "kutta": "dog insult",
            "kutti": "dog insult",
            "janwar": "animal insult",
            "jaanwar": "animal insult",
            "wahiyat": "disgusting",
            "ghatiya": "lowly",
            "tatti": "garbage",
            "ganda": "dirty",
            "gandi": "dirty",
            "gande": "dirty",
            "gandgi": "filth",
            "beizzati": "insult",
            "badtameez": "rude",
            "badtameezi": "rudeness",
            "chugalkhor": "gossiper",
            "jhootha": "liar",
            "jhooti": "liar",
            "jhooth": "lie",
            "makkar": "deceitful",
            "dhoka": "betrayal",
            "dhokha": "betrayal",
            "chomu": "fool",
            "zhandu": "fool",
            
            # ----- Positive / Neutral Informal Terms -----
            "zabardast": "amazing",
            "mast": "great",
            "mazaa": "fun",
            "maza": "fun",
            "accha": "good",
            "acchi": "good",
            "achha": "good",
            "badhiya": "excellent",
            "sahi": "right correct",
            "pakka": "confirmed definitely",
            "theek": "okay fine",
            "thik": "okay fine",
            "dost": "friend",
            "yaar": "friend",
            "yar": "friend",
            "bhai": "brother friend",
            "bro": "brother friend",
            "didi": "sister",
            "behen": "sister",
            "jaan": "dear",
            "janeman": "sweetheart",
            "bindaas": "carefree",
            
            # ----- Internet Slang / Abbreviations -----
            "lol": "laughing",
            "lmao": "laughing hard",
            "rofl": "laughing",
            "omg": "oh my god",
            "wtf": "what the heck",
            "stfu": "shut up",
            "smh": "shaking my head",
            "ngl": "not gonna lie",
            "tbh": "to be honest",
            "imo": "in my opinion",
            "bruh": "brother",
            "fam": "family",
            "sus": "suspicious",
            "salty": "upset bitter",
            "lit": "amazing exciting",
            "savage": "fierce brutal",
            "toxic": "harmful negative",
            "cringe": "embarrassing",
            "vibe": "feeling mood",
            "flex": "show off",
            "slay": "dominate impress",
            "goat": "greatest of all time",
            "cap": "lie",
            "no cap": "no lie",
            "lowkey": "slightly secretly",
            "highkey": "very obviously",
            "deadass": "seriously",
            "bet": "okay agreed",
            "simp": "overly devoted",
            "stan": "obsessive fan",
            "ratio": "disagreement",
            "L": "loss failure",
            "W": "win success",
            "fr": "for real",
            "ong": "on god",
            
            # ----- Hindi Informal Words -----
            # NOTE: Common Hindi grammar words like "hai", "hain", "kya",
            # "aur", "nahi", "tha", "thi", "aaj", "kal", "abhi", "bahut"
            # are NOT included here because:
            # 1. mBERT already understands them natively (trained on Hindi Wikipedia)
            # 2. Translating "hai" -> "is" destroys Hinglish context
            # 3. These filler words caused massive false positives in LIME
            # Only actual SLANG (non-standard informal words) are normalized below.
            "faltu": "useless waste",
            "bekaar": "useless",
            "bekar": "useless",
            "sasta": "cheap",
            "sasti": "cheap",
            "nakli": "fake",
            "asli": "real",
            
            # ----- Social Media Specific -----
            "dm": "direct message",
            "rt": "retweet",
            "irl": "in real life",
            "afaik": "as far as I know",
            "fyi": "for your information",
            "tl": "timeline",
        }
        
        # Compile a regex pattern for efficient matching
        # \b ensures we match whole words only (not substrings)
        # Sort by length (longest first) to match longer slang before shorter
        sorted_slang = sorted(self.slang_dict.keys(), key=len, reverse=True)
        escaped = [re.escape(word) for word in sorted_slang]
        self.pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )
    
    def normalize(self, text):
        """
        Replace slang terms in text with standard English equivalents.
        
        The replacement is case-insensitive and matches whole words only.
        
        Args:
            text (str): Input text with potential slang
            
        Returns:
            str: Text with slang replaced by standard equivalents
            
        Example:
            >>> normalizer = SlangNormalizer()
            >>> normalizer.normalize("yeh bakwas hai, pagal ho kya")
            'yeh nonsense hai, crazy ho what'
        """
        def replace_match(match):
            """Replace matched slang with its standard form."""
            word = match.group(0).lower()
            return self.slang_dict.get(word, word)
        
        return self.pattern.sub(replace_match, text)
    
    def get_slang_count(self, text):
        """
        Count how many slang words are in the text.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of slang words found
        """
        return len(self.pattern.findall(text))
    
    def find_slang_words(self, text):
        """
        Find all slang words in the text and their replacements.
        
        Args:
            text (str): Input text
            
        Returns:
            list[tuple]: List of (slang_word, replacement) pairs
        """
        matches = self.pattern.findall(text)
        return [(m, self.slang_dict.get(m.lower(), m)) for m in matches]
    
    def add_slang(self, slang_word, replacement):
        """
        Add a new slang word to the dictionary.
        
        Args:
            slang_word (str): The slang term
            replacement (str): The standard English replacement
        """
        self.slang_dict[slang_word.lower()] = replacement
        # Rebuild the regex pattern
        sorted_slang = sorted(self.slang_dict.keys(), key=len, reverse=True)
        escaped = [re.escape(word) for word in sorted_slang]
        self.pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )


# ============================================
# DEMONSTRATION
# ============================================
if __name__ == "__main__":
    normalizer = SlangNormalizer()
    
    test_texts = [
        "yeh bakwas hai, pagal ho kya",
        "kya ghatiya performance thi, nalayak hai",
        "bohot mast party thi bhai, zabardast",
        "chup kar bc, stfu saale",
        "ngl yeh banda lowkey toxic hai",
        "tu toh nikamma hai bilkul, kameena",
        "lol bruh that was cringe af",
    ]
    
    print("=" * 70)
    print("  SLANG NORMALIZER - DEMO")
    print("=" * 70)
    
    for text in test_texts:
        normalized = normalizer.normalize(text)
        slang_found = normalizer.find_slang_words(text)
        
        print(f"\n  Input:      {text}")
        print(f"  Normalized: {normalized}")
        print(f"  Slang:      {slang_found}")
        print("-" * 70)
