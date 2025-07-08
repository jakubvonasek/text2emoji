import csv
import os
from semantic_model import SemanticEmojiModel

class InteractiveTrainer:
    def __init__(self):
        self.model = SemanticEmojiModel()
        self.dataset_file = 'data/interactive_dataset.csv'
        self.feedback_file = 'data/user_feedback.csv'
        self.load_model()
        self.load_dataset()
        
    def load_model(self):
        """load trained model"""
        if os.path.exists('data/semantic_emoji_model.pkl'):
            self.model.load_model('data/semantic_emoji_model.pkl')
            print("✓ model loaded")
        else:
            print("✗ no model found, run semantic_trainer.py first")
            return False
        return True
    
    def load_dataset(self):
        """load existing dataset"""
        self.dataset = []
        if os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.dataset.append((row['text'], row['emoji']))
            print(f"✓ loaded {len(self.dataset)} existing examples")
    
    def save_dataset(self):
        """save dataset to csv"""
        with open(self.dataset_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'emoji'])
            for text, emoji in self.dataset:
                writer.writerow([text, emoji])
        print(f"✓ saved {len(self.dataset)} examples to {self.dataset_file}")
    
    def save_feedback(self, text, predicted, expected, feedback):
        """save user feedback"""
        with open(self.feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([text, predicted, expected, feedback])
    
    def predict_and_show(self, text):
        """predict emoji and show top 5"""
        predicted, scores = self.model.predict(text)
        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n📝 text: '{text}'")
        print(f"🤖 predicted: {predicted}")
        print("\ntop 5 predictions:")
        for i, (emoji, score) in enumerate(top_5, 1):
            print(f"  {i}. {emoji} ({score:.3f})")
        
        return predicted, top_5
    
    def get_user_feedback(self, text, predicted, top_5):
        """get user feedback on prediction"""
        print(f"\n❓ is '{predicted}' correct for '{text}'?")
        print("options:")
        print("  y = yes, correct")
        print("  n = no, wrong")
        print("  c = choose from top 5")
        print("  s = suggest different emoji")
        print("  q = quit")
        
        while True:
            choice = input("\nyour choice: ").lower().strip()
            
            if choice == 'y':
                self.dataset.append((text, predicted))
                self.save_feedback(text, predicted, predicted, 'correct')
                print("✓ added to dataset")
                return True
                
            elif choice == 'n':
                print("❌ prediction was wrong")
                return False
                
            elif choice == 'c':
                print("\nchoose from top 5:")
                for i, (emoji, score) in enumerate(top_5, 1):
                    print(f"  {i}. {emoji}")
                
                try:
                    idx = int(input("enter number (1-5): ")) - 1
                    if 0 <= idx < len(top_5):
                        chosen_emoji = top_5[idx][0]
                        self.dataset.append((text, chosen_emoji))
                        self.save_feedback(text, predicted, chosen_emoji, 'chosen_from_top5')
                        print(f"✓ added '{text}' -> '{chosen_emoji}' to dataset")
                        return True
                    else:
                        print("invalid number")
                except ValueError:
                    print("invalid input")
                    
            elif choice == 's':
                suggestion = input("enter your emoji suggestion: ").strip()
                if suggestion:
                    self.dataset.append((text, suggestion))
                    self.save_feedback(text, predicted, suggestion, 'user_suggestion')
                    print(f"✓ added '{text}' -> '{suggestion}' to dataset")
                    return True
                else:
                    print("no emoji entered")
                    
            elif choice == 'q':
                return None
                
            else:
                print("invalid choice, try again")
    
    def run(self):
        """main interactive loop"""
        print("🎯 interactive text2emoji trainer")
        print("=" * 40)
        
        if not self.load_model():
            return
        
        print(f"\ncurrent dataset: {len(self.dataset)} examples")
        print("enter text to predict emoji, or 'quit' to exit")
        
        while True:
            text = input("\n📝 enter text: ").strip()
            
            if text.lower() in ['quit', 'q', 'exit']:
                break
                
            if not text:
                continue
            
            # predict and show results
            predicted, top_5 = self.predict_and_show(text)
            
            # get user feedback
            result = self.get_user_feedback(text, predicted, top_5)
            
            if result is None:  # user quit
                break
        
        # save final dataset
        self.save_dataset()
        print(f"\n🎉 training session complete!")
        print(f"📊 total examples: {len(self.dataset)}")
        print(f"📁 dataset saved to: {self.dataset_file}")
        print(f"📝 feedback saved to: {self.feedback_file}")

if __name__ == "__main__":
    trainer = InteractiveTrainer()
    trainer.run() 