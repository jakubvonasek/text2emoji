import numpy as np
import torch
from transformers import BertTokenizer, BertModel, PreTrainedModel, PretrainedConfig
from sklearn.metrics.pairwise import cosine_similarity
import emoji
import pickle
import os

class Text2EmojiConfig(PretrainedConfig):
    model_type = "text2emoji"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Text2EmojiModel(PreTrainedModel):
    config_class = Text2EmojiConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.emoji_descriptions = {}
        self.emoji_embeddings = {}
        self.emoji_list = []
        
        # custom emoji descriptions
        self.custom_descriptions = {
            "👏": "applause clapping hands great job well done",
            "🔥": "fire hot amazing awesome lit",
            "❤️": "love heart affection romantic",
            "💯": "perfect score excellent amazing",
            "😊": "happy smile joy positive",
            "🎉": "celebration party excited happy",
            "⭐": "star excellent amazing great",
            "🚀": "rocket launch success amazing",
            "💪": "strength power muscle strong",
            "🎊": "celebration party congratulations",
            "😢": "sad crying tears unhappy",
            "😡": "angry mad furious upset",
            "💔": "broken heart sad love lost",
            "😠": "angry mad upset frustrated",
            "😔": "sad disappointed unhappy",
            "👎": "thumbs down bad dislike",
            "🌧️": "rain weather sad gloomy",
            "😤": "frustrated angry steam",
            "🤢": "sick nauseous disgusted",
            "😰": "worried anxious scared",
            "🤔": "thinking pondering unsure",
            "🤷": "shrug unsure maybe",
            "📅": "calendar date time schedule",
            "💻": "computer laptop work technology",
            "📊": "chart data analysis business",
            "📚": "books reading study learning",
            "📈": "chart graph data analysis",
            "📝": "notes writing document",
            "🗓️": "calendar planning schedule",
            "🔍": "search find investigate",
            "😴": "sleep tired rest",
            "🍕": "pizza food eating lunch",
            "🎬": "movie film entertainment",
            "🎵": "music song melody",
            "🎮": "game gaming play",
            "🏋️": "gym workout exercise",
            "👨‍🍳": "cooking chef food",
            "🚗": "car drive travel",
            "🚶": "walk walking exercise",
            "📖": "book reading story",
            "☀️": "sun sunny weather",
            "🌅": "sunset beautiful sky",
            "❄️": "snow cold winter",
            "💨": "wind windy weather",
            "🌤️": "sunny weather nice",
            "⛈️": "storm thunder lightning",
            "🌨️": "snow winter cold",
            "🌸": "flower spring beautiful",
            "☕": "coffee drink morning",
            "🥗": "salad healthy food",
            "🎂": "birthday cake celebration",
            "🍷": "wine drink alcohol",
            "🍳": "cooking breakfast food",
            "🍦": "ice cream dessert sweet",
            "🍝": "pasta food dinner",
            "🍎": "apple fruit healthy",
            "🍔": "burger food fast",
            "⏰": "time deadline urgent",
            "✅": "check success done",
            "💼": "business work office",
            "💰": "money cash rich",
            "🤝": "handshake agreement deal",
            "💬": "chat message talk",
            "🔄": "update refresh reload",
            "🐛": "bug error fix",
            "⚠️": "warning alert danger",
            "🏆": "trophy winner champion",
            "🎯": "target goal achieve",
            "✨": "sparkle magic special",
            "❌": "error wrong fail",
            "💾": "save backup data",
            "⏳": "loading waiting time",
            "📶": "signal connection wifi",
            "✈️": "plane travel vacation",
            "🏖️": "beach vacation summer",
            "🏔️": "mountain hiking nature",
            "🏙️": "city urban building",
            "🏨": "hotel accommodation",
            "📔": "passport travel document",
            "🗺️": "map travel adventure",
            "🏠": "home house return"
        }
    
    def get_embedding(self, text):
        """get bert embedding for text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def emoji_to_description(self, emoji_char):
        """convert emoji to custom semantic description"""
        if emoji_char in self.custom_descriptions:
            return self.custom_descriptions[emoji_char]
        else:
            emoji_name = emoji.demojize(emoji_char)
            description = emoji_name.replace(':', '').replace('_', ' ')
            return description
    
    def build_emoji_embeddings(self, emoji_list):
        """build embeddings for all emojis"""
        self.emoji_list = emoji_list
        
        for emoji_char in emoji_list:
            description = self.emoji_to_description(emoji_char)
            self.emoji_descriptions[emoji_char] = description
            self.emoji_embeddings[emoji_char] = self.get_embedding(description)
    
    def predict(self, text):
        """predict emoji using semantic similarity"""
        text_embedding = self.get_embedding(text)
        similarities = []
        
        for emoji_char in self.emoji_list:
            emoji_embedding = self.emoji_embeddings[emoji_char]
            similarity = cosine_similarity([text_embedding], [emoji_embedding])[0][0]
            similarities.append(similarity)
        
        # get top 5 predictions
        top_indices = np.argsort(similarities)[::-1][:5]
        scores = {}
        
        for idx in top_indices:
            emoji_char = self.emoji_list[idx]
            scores[emoji_char] = similarities[idx]
        
        predicted_emoji = self.emoji_list[top_indices[0]]
        return predicted_emoji, scores
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """forward pass for huggingface compatibility"""
        # this is a wrapper for the semantic model
        # actual prediction is done via predict() method
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs) 