"""
==========================================================
Emoji Handler Module
==========================================================

PURPOSE:
    Social media text is FULL of emojis. Traditional NLP models
    can't understand emojis, so we convert them to meaningful
    text that the model CAN understand.

HOW IT WORKS:
    1. First checks a custom dictionary of common emojis
       with simplified meanings (😂 → "laugh")
    2. Falls back to the `emoji` library for less common ones
       (it converts emoji to its official Unicode name)

EXAMPLE:
    Input:  "yeh bakwas hai 😂😡"
    Output: "yeh bakwas hai laugh anger"
"""

import emoji
import re


class EmojiHandler:
    """
    Converts emojis in text to their text descriptions.
    
    Why do we need this?
    ---------------------
    Emojis carry important sentiment information:
    - 😂 in "bakwas hai 😂" suggests mockery (offensive)
    - ❤️ in "love you ❤️" indicates positive sentiment
    - 😡 in "marr jaao 😡" amplifies hate
    
    Without conversion, the model would ignore these signals!
    """
    
    def __init__(self):
        """
        Initialize with a custom emoji-to-text mapping.
        
        We use simplified, emotion-focused words instead of
        the verbose Unicode names (e.g., "laugh" instead of
        "face with tears of joy").
        """
        # Custom mapping for the most common emojis in social media
        # These simplified forms are easier for the model to learn from
        self.custom_emoji_map = {
            # Positive emotions
            "😀": "happy",
            "😃": "happy",
            "😄": "happy",
            "😁": "grin",
            "😂": "laugh",
            "🤣": "laugh",
            "😊": "smile",
            "😇": "innocent",
            "🙂": "smile",
            "😍": "love",
            "🥰": "love",
            "😘": "kiss",
            "❤️": "love",
            "💕": "love",
            "💖": "love",
            "💗": "love",
            "💓": "love",
            "💜": "love",
            "💙": "love",
            "🧡": "love",
            "💚": "love",
            "🤍": "love",
            "😎": "cool",
            "🤗": "hug",
            "🥳": "celebration",
            "🎉": "celebration",
            "🎊": "celebration",
            "✨": "sparkle",
            "🌟": "star",
            "⭐": "star",
            "🙌": "praise",
            "👏": "clap",
            "💪": "strong",
            "👍": "good",
            "✅": "correct",
            "✌️": "peace",
            "🤝": "handshake",
            
            # Negative emotions
            "😡": "anger",
            "🤬": "anger",
            "😠": "angry",
            "😤": "frustration",
            "👿": "evil",
            "💀": "death",
            "☠️": "death",
            "🔪": "violence",
            "🩸": "blood",
            "😢": "sad",
            "😭": "crying",
            "😰": "anxiety",
            "😱": "fear",
            "🤮": "disgust",
            "🤢": "disgust",
            "💔": "heartbreak",
            "😒": "unamused",
            "😑": "expressionless",
            "🙄": "eyeroll",
            "🤦": "facepalm",
            "🤦‍♂️": "facepalm",
            "🤦‍♀️": "facepalm",
            "👎": "bad",
            "🖕": "offensive_gesture",
            "🤡": "clown",
            "🗑️": "trash",
            "😈": "evil",
            "🐍": "snake",
            "🐀": "rat",
            "🐒": "monkey",
            "🪳": "cockroach",
            "🐕": "dog",
            
            # Neutral / Context-dependent
            "🔥": "fire",
            "😐": "neutral",
            "🤔": "thinking",
            "😬": "awkward",
            "😏": "smirk",
            "👀": "eyes",
            "👋": "wave",
            "👻": "ghost",
            "🧠": "brain",
            "🔧": "tool",
            "🧪": "chemical",
            "🔫": "gun",
            "🗡️": "sword",
            "⚡": "lightning",
            "☀️": "sun",
            "🌅": "sunset",
            "📸": "camera",
            "🎵": "music",
            "🎧": "headphones",
            "🎮": "gaming",
            "🎬": "movie",
            "🍿": "popcorn",
            "🍕": "pizza",
            "🍔": "burger",
            "🍗": "chicken",
            "☕": "coffee",
            "🍳": "cooking",
            "💼": "work",
            "📝": "writing",
            "💻": "computer",
            "🏍️": "bike",
            "🚂": "train",
            "🏥": "hospital",
            "🏢": "office",
            "💍": "ring",
            "💸": "money",
            "😓": "sweat",
            "😌": "relieved",
            "😩": "tired",
            "😋": "yummy",
            "🐶": "puppy",
            "🌿": "nature",
            "🏔️": "mountain",
            "🎓": "graduation",
            "💐": "flowers",
        }
    
    def convert_emojis(self, text):
        """
        Convert all emojis in the text to their text descriptions.
        
        Process:
        1. Check each character against our custom map
        2. For unknown emojis, use the `emoji` library
        3. Clean up any extra colons from the emoji library output
        
        Args:
            text (str): Input text potentially containing emojis
            
        Returns:
            str: Text with emojis replaced by descriptive words
            
        Example:
            >>> handler = EmojiHandler()
            >>> handler.convert_emojis("yeh bakwas hai 😂😡")
            'yeh bakwas hai laugh anger'
        """
        result = []
        i = 0
        
        while i < len(text):
            # Check for multi-character emojis first (some emojis are 2+ chars)
            matched = False
            
            # Try matching longer emoji sequences first (up to 7 chars)
            for length in range(min(7, len(text) - i), 0, -1):
                substr = text[i:i + length]
                if substr in self.custom_emoji_map:
                    result.append(f" {self.custom_emoji_map[substr]} ")
                    i += length
                    matched = True
                    break
            
            if not matched:
                char = text[i]
                # Check if it's an emoji using the emoji library
                if emoji.is_emoji(char):
                    # Get the emoji name from the library
                    emoji_text = emoji.demojize(char)
                    # Clean up: remove colons and underscores
                    emoji_text = emoji_text.replace(":", "").replace("_", " ").strip()
                    result.append(f" {emoji_text} ")
                else:
                    result.append(char)
                i += 1
        
        # Clean up multiple spaces
        output = "".join(result)
        output = re.sub(r'\s+', ' ', output).strip()
        return output
    
    def has_emojis(self, text):
        """
        Check if text contains any emojis.
        
        Args:
            text (str): Input text to check
            
        Returns:
            bool: True if emojis are present
        """
        return any(emoji.is_emoji(char) for char in text)
    
    def get_emoji_count(self, text):
        """
        Count the number of emojis in text.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of emojis found
        """
        return sum(1 for char in text if emoji.is_emoji(char))


# ============================================
# DEMONSTRATION
# ============================================
if __name__ == "__main__":
    handler = EmojiHandler()
    
    # Test with various emoji-rich texts
    test_texts = [
        "yeh bakwas hai 😂😡",
        "happy birthday bhai 🎉🥳🎊",
        "I love you ❤️😘",
        "marr jaao saale 💀🔪",
        "kya mast khana tha 😋🍕",
        "so annoying yeh banda 😤🤮",
        "gym mein PR break kiya 💪🔥",
    ]
    
    print("=" * 60)
    print("  EMOJI HANDLER - DEMO")
    print("=" * 60)
    
    for text in test_texts:
        converted = handler.convert_emojis(text)
        count = handler.get_emoji_count(text)
        print(f"\n  Input:    {text}")
        print(f"  Output:   {converted}")
        print(f"  Emojis:   {count} found")
        print("-" * 60)
