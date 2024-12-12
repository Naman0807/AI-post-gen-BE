# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
from datetime import datetime, timedelta
import os
import google.generativeai as genai
from PIL import Image
import io
import base64
import random
from flask_pymongo import PyMongo
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://127.0.0.1:27017/postcraft"
app.config["JWT_SECRET_KEY"] = "1we3W4rt"
mongo = PyMongo(app)
jwt = JWTManager(app)

# Store API keys and configurations
config = {
    "huggingface_key": None,
    "gemini_key": None,
    "hf_headers": None,
    "hf_image_url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo",
    "gemini_model": None,
}

# Add this after your app initialization
try:
    mongo.db.command("ping")
    print("MongoDB connected successfully!")
except Exception as e:
    print("MongoDB connection error:", str(e))


def get_platform_specific_prompt(platform, topic, length="medium"):
    # Define word count ranges
    length_ranges = {"small": "150-300", "medium": "300-400", "long": "400-550"}
    word_range = length_ranges.get(length, "300-400")

    base_prompt = f"""Write a completely human-like, undetectable, and natural {platform} post about {topic}. 
    
Key requirements:
- Write in a conversational, human tone, use easy to read language
- Use natural language patterns and varied sentence structures
- Include personal opinions and experiences
- Add some imperfections (like starting sentences with 'And' or 'But')
- Use contractions (e.g., 'I'm' instead of 'I am')
- Vary paragraph lengths
- Include rhetorical questions
- Keep the post between {word_range} words
- Ensure content is 100% unique and plagiarism-free
- Add subtle emotional elements
- Use informal transitions between ideas

Write as if you're a real person sharing their thoughts, not an AI."""

    prompts = {
        "linkedin": f"""{base_prompt}

Additional LinkedIn-specific guidelines:
- Write from a first-person perspective
- Share a personal work experience or insight
- Include some vulnerability or learning moment
- Add natural professional enthusiasm
- Include 4-6 relevant but not overly formal hashtags
- End with a genuine question to encourage discussion
- Keep the tone professional but warm
- Avoid corporate jargon and buzzwords""",
        "twitter": f"""{base_prompt}

Additional Twitter-specific guidelines:
- Use natural Twitter language (like 'tbh', 'imo')
- Add personality and character
- Include 3-5 relevant hashtags that feel natural
- Write as if you're sharing a quick thought
- Add some personality quirks""",
        "instagram": f"""{base_prompt}

Additional Instagram-specific guidelines:
- Write in a casual, friendly tone
- Include personal feelings and reactions
- Use a natural mix of emojis (not too many)
- Add 3-5 relevant hashtags that feel organic
- Share a moment or feeling
- Make it feel like a friend's post""",
    }
    return prompts.get(platform)


def get_platform_specific_image_prompt(platform, topic):
    prompts = {
        "linkedin": f"""Create a LinkedIn Marketing Image about {topic}:

Style: Clean, corporate, and modern
Color Scheme: Blues, grays, with subtle accent colors (e.g., white, metallics)
Text:
Headline: Max 6-8 words, short and impactful
Additional Text: Minimal, only key message
Font: Bold, modern, high contrast
Imagery:
High-quality, professional, and relevant to the topic
Avoid generic stock photos
Contrast: Ensure high contrast for easy readability (mobile and desktop)
Layout: Clear, organized, with a focus on the headline
Goal: Capture attention, look professional, and drive engagement without clutter.""",
        "twitter": f"""Create a striking social media image for Twitter about {topic}.
Requirements:
- Bold, attention-grabbing design
- High contrast colors
- Maximum one short phrase
- Simple enough to be visible on mobile
- Memorable visual impact""",
        "instagram": f"""Create a visually stunning image for Instagram about {topic}.
Requirements:
- Beautiful, artistic composition
- Instagram-optimized square format
- Vibrant but cohesive colors
- Minimal text if any
- Highly visual focus
- Memorable and shareable design""",
    }
    return prompts.get(platform)


@app.route("/auth/register", methods=["POST"])
def register():
    try:
        data = request.json
        print(
            "Received registration data:", {**data, "password": "*****"}
        )  # Log data safely

        # Validate required fields
        required_fields = ["name", "email", "password"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Validate email format
        if "@" not in data["email"]:
            return jsonify({"error": "Invalid email format"}), 400

        # Check if email already exists
        existing_user = mongo.db.users.find_one({"email": data["email"]})
        if existing_user:
            return jsonify({"error": "Email already registered"}), 400

        # Create new user
        new_user = {
            "name": data["name"],
            "email": data["email"],
            "password": generate_password_hash(data["password"]),
            "posts": [],
            "hf_api_key": None,
            "gemini_api_key": None,
            "setup_completed": False,
            "created_at": datetime.utcnow(),
        }

        # Insert user into database
        result = mongo.db.users.insert_one(new_user)

        # Generate JWT token
        token = create_access_token(identity=str(result.inserted_id))

        return (
            jsonify(
                {
                    "token": token,
                    "user": {
                        "id": str(result.inserted_id),
                        "name": new_user["name"],
                        "email": new_user["email"],
                    },
                }
            ),
            201,
        )

    except Exception as e:
        print("Registration error:", str(e))  # Log the error
        return jsonify({"error": "Registration failed: " + str(e)}), 500


@app.route("/auth/login", methods=["POST"])
def login():
    try:
        data = request.json

        if not data.get("email") or not data.get("password"):
            return jsonify({"error": "Email and password are required"}), 400

        user = mongo.db.users.find_one({"email": data["email"]})

        if not user or not check_password_hash(user["password"], data["password"]):
            return jsonify({"error": "Invalid email or password"}), 401

        token = create_access_token(
            identity=str(user["_id"]), expires_delta=timedelta(days=7)
        )

        return (
            jsonify(
                {
                    "token": token,
                    "user": {
                        "id": str(user["_id"]),
                        "name": user["name"],
                        "email": user["email"],
                    },
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/initialize", methods=["POST"])
@jwt_required()
def initialize_apis():
    try:
        data = request.json
        hf_api_key = data.get("hf_api_key")
        gemini_api_key = data.get("gemini_api_key")

        if not hf_api_key or not gemini_api_key:
            return jsonify({"error": "Both API keys are required"}), 400

        # Initialize Hugging Face headers
        config["hf_headers"] = {"Authorization": f"Bearer {hf_api_key}"}
        config["hf_image_url"] = (
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo"
        )

        # Initialize Gemini model
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)
        config["gemini_model"] = genai.GenerativeModel("gemini-1.5-flash")

        # Test both APIs to ensure they work
        try:
            # Test Gemini
            response = config["gemini_model"].generate_content("Test message")
            if not response:
                raise Exception("Failed to initialize Gemini API")

            # Test Hugging Face
            test_response = requests.post(
                config["hf_image_url"],
                headers=config["hf_headers"],
                json={"inputs": "test"},
            )
            if test_response.status_code not in [
                200,
                503,
            ]:  # 503 is acceptable as it means model is loading
                raise Exception("Failed to initialize Hugging Face API")

        except Exception as e:
            return jsonify({"error": f"API test failed: {str(e)}"}), 500

        return jsonify({"message": "APIs initialized successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Initialization failed: {str(e)}"}), 500


def humanize_content(text, platform):
    """Post-process the generated content to make it more human-like"""
    try:
        # Add natural variations and imperfections
        variations = {
            "definitely": ["def", "definitely", "for sure"],
            "amazing": ["amazing", "awesome", "fantastic", "great"],
            "think": ["think", "believe", "feel like"],
            "very": ["very", "really", "pretty", "quite"],
        }

        # Replace some words with their variations randomly
        for word, alternatives in variations.items():
            if word in text.lower():
                text = text.replace(word, random.choice(alternatives))

        # Add some natural pauses and flow
        text = text.replace(". ", "... ")  # Occasionally add ellipsis
        text = text.replace(
            "!", random.choice(["!", "!!", "! "])
        )  # Vary exclamation marks

        # Platform-specific humanization
        if platform == "twitter":
            # Make it more Twitter-like
            text = text.replace("because", "bc")
            text = text.replace("with", "w/")
            text = text.replace("without", "w/o")

        # Clean up any artificial patterns
        text = text.replace("  ", " ")  # Remove double spaces
        text = text.replace("...", "...")  # Standardize ellipsis
        text = text.replace("!!", "!")  # Clean up multiple exclamations

        return text
    except Exception as e:
        print(f"Error in humanize_content: {str(e)}")
        return text  # Return original text if processing fails


@app.route("/generate_post", methods=["POST"])
@jwt_required()  # Requires authentication
def generate_post():
    if not config.get("gemini_model") or not config.get("hf_headers"):
        return (
            jsonify({"error": "APIs not initialized. Please initialize APIs first"}),
            400,
        )

    try:
        user_id = get_jwt_identity()
        data = request.json
        topic = data.get("topic")
        platform = data.get("platform")
        image_count = data.get("imageCount", 1)
        post_length = data.get("postLength", "medium")
        temperature = data.get("temperature", 0.7)

        if not topic or not platform:
            return jsonify({"error": "Topic and platform are required"}), 400

        # Your existing text and image generation code here...
        text_prompt = get_platform_specific_prompt(platform, topic, post_length)
        base_image_prompt = get_platform_specific_image_prompt(platform, topic)

        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,  # Slightly increase diversity
            "top_k": 50,  # Increase sampling pool
        }

        # Generate multiple versions and pick the most human-like one
        attempts = 3
        best_content = None
        lowest_ai_score = float("inf")

        for _ in range(attempts):
            response = config["gemini_model"].generate_content(
                text_prompt, generation_config=generation_config
            )
            content = response.text.strip()
            content = content.replace("**", "").replace("#", " #")

            # Humanize the content
            humanized_content = humanize_content(content, platform)

            # Here you could add an AI detection check if you have an API for it
            # For now, we'll use the last generated version
            best_content = humanized_content

        if platform == "twitter" and len(best_content) > 280:
            best_content = best_content[:277] + "..."

        # Generate images (your existing image generation code)
        images = []
        variations = [
            "from a different perspective",
            "with a modern style",
            "with a creative layout",
            "with a minimalist design",
        ]

        for i in range(image_count):
            variation = variations[i % len(variations)]
            modified_prompt = f"{base_image_prompt} {variation}"

            image_response = requests.post(
                config["hf_image_url"],
                headers=config["hf_headers"],
                json={
                    "inputs": modified_prompt,
                    "negative_prompt": "duplicate, similar images, same composition",
                    "seed": random.randint(1, 999999),
                },
            )

            if image_response.status_code != 200:
                raise Exception(f"Image generation failed: {image_response.text}")

            image_data = base64.b64encode(image_response.content).decode("utf-8")
            images.append(f"data:image/jpeg;base64,{image_data}")

        # After generating the content, calculate engagement score
        engagement_prompt = f"""
        Analyze this social media post for {platform} and give it an engagement score out of 100.
        Consider factors like:
        - Readability
        - Call to action
        - Emotional appeal
        - Hashtag usage
        - Length appropriateness for the platform
        
        Post: {best_content}
        
        Return only the numeric score.
        """

        score_response = config["gemini_model"].generate_content(engagement_prompt)
        engagement_score = int(score_response.text.strip())

        # Save post to user's posts in MongoDB
        post = {
            "platform": platform,
            "topic": topic,
            "content": best_content,
            "images": images,
            "engagement_score": engagement_score,
            "created_at": datetime.utcnow(),
        }

        mongo.db.users.update_one(
            {"_id": ObjectId(user_id)}, {"$push": {"posts": post}}
        )

        return (
            jsonify(
                {
                    "text": best_content,
                    "images": images,
                    "engagement_score": engagement_score,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/user/posts", methods=["GET"])
@jwt_required()
def get_user_posts():
    try:
        user_id = get_jwt_identity()
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

        if not user:
            return jsonify({"error": "User not found"}), 404

        posts = user.get("posts", [])
        for post in posts:
            post["_id"] = str(post["_id"])
            if "engagement_score" not in post:
                post["engagement_score"] = 0
        return jsonify({"posts": posts}), 200

    except Exception as e:
        print(f"Error in get_user_posts: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/user/analytics", methods=["GET"])
@jwt_required()
def get_user_analytics():
    try:
        user_id = get_jwt_identity()
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

        # Calculate analytics
        platform_stats = {}
        total_posts = len(user["posts"])

        for post in user["posts"]:
            platform = post["platform"]
            if platform not in platform_stats:
                platform_stats[platform] = 0
            platform_stats[platform] += 1

        return (
            jsonify(
                {
                    "total_posts": total_posts,
                    "platform_distribution": platform_stats,
                    "recent_posts": user["posts"][-5:],
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
