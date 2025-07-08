# Text2Emoji Model

A semantic emoji prediction model using BERT embeddings and cosine similarity matching.

## Model Description

This model predicts appropriate emojis for text input using semantic similarity between text embeddings and emoji descriptions. It uses BERT-base-uncased for text encoding and cosine similarity for matching.

### Model Type
- **Architecture**: BERT-base-uncased + semantic similarity
- **Task**: Text-to-emoji prediction
- **Language**: English
- **License**: MIT

## Usage

```python
from transformers import AutoTokenizer, AutoModel
from model import Text2EmojiModel
import pickle

# load the model
config = Text2EmojiConfig()
model = Text2EmojiModel(config)

# load trained emoji embeddings
with open('semantic_emoji_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model.emoji_descriptions = model_data['emoji_descriptions']
    model.emoji_embeddings = model_data['emoji_embeddings']
    model.emoji_list = model_data['emoji_list']

# predict emoji
text = "I'm feeling happy today!"
emoji, confidence = model.predict_emoji(text)
print(f"Text: {text}")
print(f"Predicted emoji: {emoji}")
print(f"Confidence: {confidence:.3f}")
```

## Training Data

- **Dataset size**: 357 text-emoji pairs
- **Source**: Custom curated dataset with diverse contexts
- **Examples**:
  - "good morning" → ☕
  - "beautiful weather" → 🌤️
  - "love you" → ❤️
  - "sleep tight" → 😴

## Performance

- **Dataset size**: 357 pairs (from data.json)
- **Search-style accuracy (top 3)**: 15.1% (54/357)
- **Model file**: data/semantic_emoji_model.pkl
- **Predictions CSV**: data/predictions.csv

**Notes**: More data and better emoji descriptions will improve results.

## Evaluation Results

### Test Examples
- ✅ "beautiful weather today" → 🌤️ (good!)
- ✅ "grabbing lunch" → 🍳 (reasonable)
- ❌ "coding all night" → 📖 (should be 💻)
- ❌ "stressed about deadlines" → 😰 (good emotion match)

### Common Issues
1. **Generic predictions**: Many inputs default to 💒, 🎓, 🎊
2. **Limited semantic matching**: Model struggles with specific contexts
3. **Need better emoji descriptions**: Current descriptions may not capture full meaning

## Model Architecture

### Components
1. **BERT Encoder**: bert-base-uncased for text embedding
2. **Emoji Descriptions**: Custom descriptions for 73 emojis
3. **Similarity Matching**: Cosine similarity between text and emoji embeddings
4. **Prediction**: Top-k emoji selection based on similarity scores

### Emoji Coverage
The model supports 73 emojis including:
- Emotions: 😊, 😴, 😰, ❤️, 💔
- Weather: 🌤️, ☀️, 🌙, 🌨️
- Activities: 🍳, ☕, 📝, 💻
- Objects: 🏠, 🚗, ✈️, 🎓

## Limitations

1. **Accuracy**: Current accuracy is limited (15.1% top-3)
2. **Context sensitivity**: May miss nuanced meanings
3. **Emoji coverage**: Limited to 73 emojis
4. **Cultural bias**: Trained on English text
5. **Ambiguity**: Same text can have multiple valid emoji interpretations

## Training Details

### Data Processing
- Text preprocessing using BERT tokenizer
- Custom emoji descriptions for semantic matching
- Cosine similarity for embedding comparison

### Model Files
- `model.py`: Hugging Face compatible model wrapper
- `config.json`: Model configuration
- `semantic_emoji_model.pkl`: Trained embeddings
- `tokenizer_config.json`: Tokenizer settings

## Usage Examples

```python
# Basic prediction
text = "I'm excited about the weekend!"
emoji, confidence = model.predict_emoji(text)
# Output: 🎉 (0.711)

# Top 3 predictions
text = "stressed about deadlines"
top_emojis = model.predict_top_k(text, k=3)
# Output: [😰, 😤, ⏰]

# Interactive training
from src.interactive_trainer import InteractiveTrainer
trainer = InteractiveTrainer()
trainer.run()  # Interactive training session
```

## Citation

```bibtex
@misc{text2emoji2025,
  title={Text2Emoji: Semantic Emoji Prediction using BERT},
  author={Jakub Vonasek},
  year={2025},
  url={https://huggingface.co/jakubvonasek/text2emoji}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Areas for improvement:
- Better emoji descriptions
- More training data
- Improved model architecture
- Evaluation metrics

## Contact

- Model: [jakubvonasek/text2emoji](https://huggingface.co/jakubvonasek/text2emoji)
- Issues: Report via Hugging Face discussions
