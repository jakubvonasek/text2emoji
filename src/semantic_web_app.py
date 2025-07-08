from flask import Flask, render_template, request, jsonify
import pickle
import os
from semantic_model import SemanticEmojiModel

app = Flask(__name__)

# load semantic model
def load_semantic_model():
    if os.path.exists('semantic_emoji_model.pkl'):
        model = SemanticEmojiModel()
        model.load_model('semantic_emoji_model.pkl')
        return model
    return None

@app.route('/')
def home():
    return render_template('semantic_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    model = load_semantic_model()
    
    if model and text:
        predicted_emoji, scores = model.predict(text)
        # get top 5 predictions
        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        # convert all scores to float for json serialization
        top_5 = [(k, float(v)) for k, v in top_5]
        return jsonify({
            'prediction': predicted_emoji,
            'scores': dict(top_5)
        })
    
    return jsonify({'error': 'semantic model not found or invalid input'})

if __name__ == '__main__':
    app.run(debug=True, port=5001) 