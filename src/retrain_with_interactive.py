import csv
import os
from semantic_model import SemanticEmojiModel
from semantic_trainer import TRAINING_DATA

def load_interactive_dataset(filename='data/interactive_dataset.csv'):
    """load dataset from interactive training"""
    dataset = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset.append((row['text'], row['emoji']))
        print(f"✓ loaded {len(dataset)} interactive examples")
    return dataset

def combine_datasets(original_data, interactive_data):
    """combine original and interactive datasets"""
    combined = original_data + interactive_data
    print(f"✓ combined dataset: {len(combined)} total examples")
    return combined

def retrain_model(combined_data, save_path='data/semantic_emoji_model.pkl'):
    """retrain model with combined dataset"""
    print("🔄 retraining model...")
    
    # split data
    texts = [item[0] for item in combined_data]
    emojis = [item[1] for item in combined_data]
    
    # train model
    model = SemanticEmojiModel()
    model.train(texts, emojis)
    
    # save model
    model.save_model(save_path)
    print(f"✓ model saved to {save_path}")
    
    return model

def main():
    print("🔄 retrain with interactive dataset")
    print("=" * 40)
    
    # load original training data
    original_data = TRAINING_DATA
    print(f"✓ original dataset: {len(original_data)} examples")
    
    # load interactive dataset
    interactive_data = load_interactive_dataset()
    
    if not interactive_data:
        print("❌ no interactive dataset found")
        print("run interactive_trainer.py first")
        return
    
    # combine datasets
    combined_data = combine_datasets(original_data, interactive_data)
    
    # retrain model
    model = retrain_model(combined_data)
    
    print("\n🎉 retraining complete!")
    print(f"📊 new model trained on {len(combined_data)} examples")
    print("you can now use the improved model in the web app")

if __name__ == "__main__":
    main() 