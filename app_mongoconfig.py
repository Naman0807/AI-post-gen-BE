from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import base64
import random
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import requests
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


load_dotenv()
mongo_uri = os.getenv("ATLAS_URI")

# Initialize Flask app
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                os.getenv("FE_URL"),
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type", "Authorization"],
        }
    },
    supports_credentials=True,
)

# JWT Configuration
app.config["JWT_SECRET_KEY"] = "1we3W4rt"

# Initialize JWT
jwt = JWTManager(app)

try:
    client = MongoClient(mongo_uri, server_api=ServerApi("1"))
    db = client.postcraft
    client.admin.command("ping")
    print("\n\n\n\t\tMongoDB Atlas connection successful!\n\n\n")
except Exception as e:
    print("MongoDB Atlas connection error:", str(e))
    raise e


app_config = {
    "huggingface_key": None,
    "gemini_key": None,
    "hf_headers": None,
    "hf_image_url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo",
    "gemini_model": None,
}


def get_platform_specific_prompt(platform, topic, length="medium"):
    # Define word count ranges
    length_ranges = {"small": "50-100", "medium": "100-200", "long": "200-300"}
    word_range = length_ranges.get(length, "100-200")

    base_prompt = f"""Write a natural, human-like, and completely unique {platform} post about {topic} that feels authentic and conversational.
Here’s what to include:
Tone & Style: Write casually and conversationally, as if you're talking to a friend. Use natural language, with varied sentence structures and phrasing.
Personal Touch: Include personal opinions, small anecdotes, or experiences to make it relatable.
Imperfections: Add small imperfections like starting sentences with "And" or "But," using contractions (e.g., "I'm," "you're"), and breaking long ideas into shorter, punchy sentences.
Paragraph Flow: Vary the lengths of paragraphs. Some should be just one or two lines, others a little longer.
Engaging Questions: Sprinkle in rhetorical or reflective questions to draw the reader in and keep it dynamic.
Emotional Subtlety: Infuse a bit of emotion, whether it’s curiosity, excitement, humor, or even mild frustration, to make it feel alive.
Transitions: Use informal transitions like "Anyway," "So," or "Honestly" to keep it flowing smoothly, as you’d do in a casual conversation.
Length: Keep the post between {word_range} words, making every word feel purposeful but not overly polished.
Ultimately, the goal is to write like a real person sharing honest thoughts—not a polished, overly structured essay."""

    prompts = {
        "linkedin": f"""{base_prompt}

- Write in First Person: Share authentically from your perspective.
- Tell a Story: Highlight a work experience or lesson learned.
- Show Vulnerability: Share challenges or growth moments.
- Be Enthusiastic: Let your passion shine naturally.
- Use 4-6 Hashtags: Keep them relevant and simple.
- End with a Question: Spark engagement with an open-ended question.
- Be Warm, Not Jargony: Stay professional but conversational.
""",
        "twitter": f"""{base_prompt}

- Additional Twitter-specific guidelines:
- Keep it casual and conversational (use stuff like "tbh," "imo," or emojis for extra vibe).
- Let your personality shine—make it feel like you’re talking, not a bot.
- Throw in 3-5 relevant hashtags, but make them flow naturally (no hashtag spam).
- Keep it short and punchy—like a quick thought or reaction you’d share with a friend.
- Add quirks! Whether it’s humor, sass, or a unique POV, make it memorable.""",
    }
    return prompts.get(platform)


def get_platform_specific_image_prompt(platform, topic):
    prompts = {
        "linkedin": f"""Create a image about {topic} for linkedin:
- Style: Clean, corporate, modern
- Headline: 6-8 words, short and impactful
- Additional Text: Minimal, key message only
- Font: Bold, modern, high contrast
- Imagery: High-quality, topic-relevant (no generic stock photos)
- Contrast: High-contrast for readability (mobile & desktop)
- Layout: Clear, organized, headline-focused
- Goal: Professional, attention-grabbing, clutter-free, engaging""",
        "twitter": f"""Create a visually stunning Twitter image about {topic}.
Requirements:
- Bold, eye-catching design that stops the scroll
- High-contrast color palette for maximum visibility
- One impactful, short phrase (5 words max)
- Clean, minimalist layout optimized for mobile
- Strong, memorable visuals that leave a lasting impression""",
    }
    return prompts.get(platform)


@app.route("/auth/register", methods=["POST"])
def register():
    try:
        data = request.json
        print("Received registration data:", {**data, "password": "*****"})

        # Validate required fields
        required_fields = ["name", "email", "password"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Validate email format
        if "@" not in data["email"]:
            return jsonify({"error": "Invalid email format"}), 400

        # Check if email already exists
        existing_user = db.users.find_one({"email": data["email"]})
        if existing_user:
            return jsonify({"error": "Email already registered"}), 400

        # Create new user
        new_user = {
            "name": data["name"],
            "email": data["email"],
            "password": generate_password_hash(
                data["password"], method="pbkdf2:sha256"
            ),
            "posts": [],
            "hf_api_key": None,
            "gemini_api_key": None,
            "setup_completed": False,
            "created_at": datetime.utcnow(),
        }

        # Insert user into database
        result = db.users.insert_one(new_user)

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
        print("Registration error:", str(e))
        return jsonify({"error": "Registration failed: " + str(e)}), 500


@app.route("/auth/login", methods=["POST"])
def login():
    try:
        data = request.json

        if not data.get("email") or not data.get("password"):
            return jsonify({"error": "Email and password are required"}), 400

        user = db.users.find_one({"email": data["email"]})

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


# Then modify your initialize_apis function
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
        app_config["hf_headers"] = {"Authorization": f"Bearer {hf_api_key}"}
        app_config["hf_image_url"] = (
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo"
        )

        # Initialize Gemini model
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)
        app_config["gemini_model"] = genai.GenerativeModel("gemini-1.5-flash")

        # Test both APIs to ensure they work
        try:
            # Test Gemini
            response = app_config["gemini_model"].generate_content("Test message")
            if not response:
                raise Exception("Failed to initialize Gemini API")

            # Test Hugging Face
            test_response = requests.post(
                app_config["hf_image_url"],
                headers=app_config["hf_headers"],
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
@jwt_required()
def generate_post():
    if not app_config.get("gemini_model") or not app_config.get("hf_headers"):
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
            response = app_config["gemini_model"].generate_content(
                text_prompt, generation_config=generation_config
            )
            content = response.text.strip()
            content = content.replace("**", "").replace("#", " #")

            # Humanize the content
            humanized_content = humanize_content(content, platform)
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
                app_config["hf_image_url"],
                headers=app_config["hf_headers"],
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
        Analyze this social media post for {platform} and provide an engagement score out of 100. Consider the following factors:

- Readability: Is the post easy to read and understand?
- Call to Action: Does it effectively encourage user interaction or response?
- Emotional Appeal: Does the post resonate emotionally with the audience?
- Hashtag Usage: Are hashtags used effectively and relevantly?
- Length Appropriateness: Is the post the right length for the platform (concise yet impactful)?

Post: {best_content}

Return only the numeric engagement score.
        """

        score_response = app_config["gemini_model"].generate_content(engagement_prompt)
        engagement_score = int(score_response.text.strip())

        # Save post to database
        post = {
            "_id": ObjectId(),
            "platform": platform,
            "topic": topic,
            "content": best_content,
            "images": images,
            "engagement_score": engagement_score,
            "created_at": datetime.utcnow(),
            "metadata": {
                "length": post_length,
                "temperature": temperature,
                "image_count": len(images),
            },
        }

        db.users.update_one({"_id": ObjectId(user_id)}, {"$push": {"posts": post}})

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
        user = db.users.find_one({"_id": ObjectId(user_id)})

        if not user:
            return jsonify({"error": "User not found"}), 404

        posts = user.get("posts", [])
        for post in posts:
            post["_id"] = str(post["_id"])

        return jsonify({"posts": posts}), 200

    except Exception as e:
        print(f"Error in get_user_posts: " + e)
        return jsonify({"error": str(e)}), 500


@app.route("/user/posts/<post_id>", methods=["DELETE"])
@jwt_required()
def delete_post(post_id):
    try:
        if not ObjectId.is_valid(post_id):
            return jsonify({"error": "Invalid post ID format"}), 400

        current_user_id = get_jwt_identity()
        result = db.users.update_one(
            {"_id": ObjectId(current_user_id)},
            {"$pull": {"posts": {"_id": ObjectId(post_id)}}},
        )

        if result.modified_count > 0:
            return jsonify({"message": "Post deleted successfully"}), 200
        else:
            return jsonify({"error": "Post not found or unauthorized"}), 404

    except Exception as e:
        print(f"Delete error: {str(e)}")
        return jsonify({"error": "Server error occurred"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
