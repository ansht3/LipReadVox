#!/usr/bin/env python3
"""
Demo script to collect training data for lip reading
This will help you get started with collecting data for a few basic words
"""

import sys
from pathlib import Path
sys.path.append('src')

from collection import LipReadingCollector

def main():
    print("🎤 Lip Reading Data Collection Demo")
    print("=" * 50)
    
    # Define some basic words to collect
    words = ["hello", "goodbye", "yes", "no", "thank"]
    
    print("This demo will help you collect data for basic words:")
    for i, word in enumerate(words, 1):
        print(f"   {i}. {word}")
    
    print("\n📋 Instructions:")
    print("1. For each word, you'll record multiple takes")
    print("2. Press 'R' to start recording")
    print("3. Say the word clearly while looking at the camera")
    print("4. Press 'S' to stop recording")
    print("5. Repeat 3-5 times for each word")
    print("6. Press 'Q' to quit")
    
    input("\nPress Enter to start...")
    
    for word in words:
        print(f"\n🎬 Recording data for word: '{word}'")
        print("Press Enter when ready to start...")
        input()
        
        collector = LipReadingCollector(word)
        collector.run()
        
        print(f"\n✅ Completed recording for '{word}'")
        print("Press Enter to continue to next word...")
        input()
    
    print("\n🎉 Data collection complete!")
    print("Next steps:")
    print("1. Run: python src/preprocess_training.py")
    print("2. Run: python src/model_training.py")
    print("3. Run: python src/predict.py")

if __name__ == "__main__":
    main() 