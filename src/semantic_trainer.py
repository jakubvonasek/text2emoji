from semantic_model import SemanticEmojiModel
import pickle
import numpy as np
import json
from collections import defaultdict, Counter
import math
import csv
import os

def load_training_data_from_json(filename='data/data.json'):
    """load training data from json file"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # extract text-emoji pairs from json
    training_data = []
    for item in data:
        text = item['text']
        emoji = item['emoji']
        training_data.append((text, emoji))
    
    print(f"loaded {len(training_data)} training examples from {filename}")
    return training_data

def evaluate_model(model, test_data):
    """evaluate semantic model accuracy (search engine style: correct if in top 3)"""
    correct = 0
    total = len(test_data)
    
    for text, expected_emoji in test_data:
        predicted_emoji, scores = model.predict(text)
        # get top 3 predictions
        top_3 = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]]
        if expected_emoji in top_3:
            correct += 1
            print(f"✓ '{text}' -> expected: {expected_emoji}, got: {predicted_emoji}, top3: {top_3}")
        else:
            print(f"✗ '{text}' -> expected: {expected_emoji}, got: {predicted_emoji}, top3: {top_3}")
        
        # show all top 5 predictions
        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for emoji, score in top_5:
            print(f"    {emoji}: {score:.3f}")
        print()  # empty line for readability
    
    accuracy = correct / total
    print(f"\nsearch-style accuracy (in top 3): {accuracy:.2%} ({correct}/{total})")
    return accuracy

def export_predictions_to_csv(test_data, model, filename='predictions.csv'):
    """export all predictions to csv for analysis and correction"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'expected_emoji', 'predicted_emoji', 'top1_score', 'top2_emoji', 'top2_score', 'top3_emoji', 'top3_score', 'top4_emoji', 'top4_score', 'top5_emoji', 'top5_score'])
        
        for text, expected_emoji in test_data:
            predicted_emoji, scores = model.predict(text)
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            row = [text, expected_emoji, predicted_emoji]
            for i in range(5):
                if i < len(top_5):
                    emoji, score = top_5[i]
                    row.extend([emoji, f"{score:.3f}"])
                else:
                    row.extend(['', ''])
            
            writer.writerow(row)
    
    print(f"predictions exported to {filename}")

def main():
    print("training semantic emoji prediction model...")
    
    # load training data
    training_data = load_training_data_from_json()
    
    # split data
    texts = [item[0] for item in training_data]
    emojis = [item[1] for item in training_data]
    
    split_idx = int(0.8 * len(training_data))
    train_texts = texts[:split_idx]
    train_emojis = emojis[:split_idx]
    test_data = training_data[split_idx:]
    
    # train semantic model
    model = SemanticEmojiModel()
    model.train(train_texts, train_emojis)
    
    # save model
    model.save_model('data/semantic_emoji_model.pkl')
    print("semantic model saved to data/semantic_emoji_model.pkl")
    
    # evaluate
    print("\nevaluating semantic model...")
    evaluate_model(model, test_data)
    
    # export predictions to csv
    export_predictions_to_csv(test_data, model, 'data/predictions.csv')
    
    # test with new examples
    print("\ntesting with new examples:")
    test_examples = [
        "excited about the weekend",
        "stressed about deadlines", 
        "grabbing lunch",
        "coding all night",
        "beautiful weather today"
    ]
    
    for example in test_examples:
        predicted, scores = model.predict(example)
        print(f"'{example}' -> {predicted}")
        # show top 5 predictions
        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for emoji, score in top_5:
            print(f"  {emoji}: {score:.3f}")
        print()  # empty line for readability

if __name__ == "__main__":
    main() 