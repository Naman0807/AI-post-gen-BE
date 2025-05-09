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
import google.generativeai as genai

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("ATLAS_URI")
print(os.getenv("FE_URL"))

# Initialize Flask app
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": ["https://postcraft-lab.vercel.app", "http://localhost:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "ngrok-skip-browser-warning",
            ],
            "expose_headers": ["Content-Type", "Authorization"],
        }
    },
    supports_credentials=True,
)

# JWT Configuration
app.config["JWT_SECRET_KEY"] = "1we3W4rt"

# Initialize JWT
jwt = JWTManager(app)

# MongoDB Connection
try:
    client = MongoClient(mongo_uri, server_api=ServerApi("1"))
    db = client.postcraft
    client.admin.command("ping")
    print("\n\n\n\t\tMongoDB Atlas connection successful!\n\n\n")
except Exception as e:
    print("MongoDB Atlas connection error:", str(e))
    raise e

# Global app configuration for API keys
app_config = {
    "huggingface_key": None,
    "gemini_key": None,
    "hf_headers": None,
    "hf_image_url": "https://api-inference.huggingface.co/models/strangerzonehf/Flux-Midjourney-Mix2-LoRA",
    "gemini_model": None,
}


def initialize_api_keys():
    """
    Fetch API keys from the database and initialize Gemini and Hugging Face APIs.
    Returns True if successful, False otherwise.
    """
    try:
        # Find a user with API keys (e.g., the first user who completed setup)
        user = db.users.find_one({"setup_completed": True})
        if user and user.get("hf_api_key") and user.get("gemini_api_key"):
            # Store the keys in app_config
            app_config["huggingface_key"] = user["hf_api_key"]
            app_config["gemini_key"] = user["gemini_api_key"]

            # Initialize Gemini API
            genai.configure(api_key=app_config["gemini_key"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            app_config["gemini_model"] = model

            # Initialize Hugging Face headers
            app_config["hf_headers"] = {
                "Authorization": f"Bearer {app_config['huggingface_key']}"
            }

            print("API keys successfully initialized from database")
            return True
        else:
            print("No user with API keys found for initialization")
            return False
    except Exception as e:
        print(f"Failed to initialize API keys: {e}")
        return False


def ensure_api_keys_initialized():
    """
    Ensure that API keys are initialized before proceeding.
    If not initialized, fetch them from the database.
    """
    if app_config["gemini_model"] is None or app_config["hf_headers"] is None:
        print("API keys not initialized. Attempting to fetch from database...")
        success = initialize_api_keys()
        if not success:
            raise Exception("Failed to initialize API keys from database")


# Initialize API keys at startup
with app.app_context():
    initialize_api_keys()


def get_platform_specific_prompt(platform, topic, length=200):
    word_range = f"{length-20}-{length+20}"
    base_prompt = f"""Create an authentic {platform} post about {topic} that feels like it was written in a single, natural moment. The post should:

VOICE & STRUCTURE
- Capture a distinct point of view and personality
- Flow like natural speech while maintaining clear purpose
- Vary rhythm between longer insights and punchy observations
- Use organic transitions that emerge from the ideas
- Hit approximately {word_range} words while staying natural

CONTENT APPROACH
- Share a specific perspective or experience that sparked your thinking
- Weave in concrete examples that illustrate your points
- Build connections between observations
- End with an insight that feels earned, not forced
- Optional: Incorporate relevant data points or expert perspectives if they fit naturally

WRITING STYLE
- Write with your authentic voice, not a formula
- Mix up sentence patterns - some complex, some direct
- Include natural pauses where you'd take a breath
- Make the paragraph length very (if there is any)
- Use industry terminology naturally, not to show off
- Let personality shine through while staying professional

AVOID
- Generic openings ("Here's the thing about...", "I've been thinking...")
- Overused transition phrases ("That being said", "On the other hand") 
- Forced engagement bait ("Who else agrees?", "Let me know in the comments")
- Unnecessary buzzwords and jargon
- Perfectly polished corporate speak

The goal is to create content that reads like it came from a real person sharing genuine thoughts, not following a template. Each post should have its own unique personality and flow while delivering valuable insights about {topic}.

Additional Context:
- Platform context: Adapt tone and style for {platform}'s specific audience and format
- Topic focus: Demonstrate genuine knowledge/interest in {topic}
- Authenticity: Write from a place of real experience or understanding
- Natural expertise: Share insights without trying to prove authority

**Note**
- dont want any type of extra text other then the post contnet.

here is the example output:
<example>
Okay, so I've been diving deep into Recurrent Neural Networks lately, and honestly, it's been a wild ride! I remember the first time I tried to wrap my head around LSTMs – pure brain-melt... It felt like trying to understand quantum physics while simultaneously juggling flaming torches... 

But, seriously, the power of RNNs is incredible... I was working on a project predicting customer churn, and using an RNN made a HUGE difference... The accuracy jumped, like, significantly! It was so satisfying to see the model actually *learning* patterns over time, not just spitting out random guesses.

And the best part? The feeling of finally "getting it." That moment when the complex equations suddenly clicked, and I could actually visualize how the network was processing sequential data? Pure magic... It's addictive, I'll admit it.

Anyway, I'm still learning, of course... There are always new challenges, new architectures to explore... It's a constant learning curve, which, let's be honest, can be frustrating sometimes... But that's part of the fun, right?

So, what's your favorite application of RNNs? I'd love to hear what you're working on!
</example>
"""

    prompts = {
        "linkedin": f"""{base_prompt}

LinkedIn-Specific Elements:
- Professional yet approachable tone
- Industry-relevant insights
- Strategic use of 3-4 relevant hashtags
- Clear value proposition or takeaway
- End with an engaging question or call for discussion
- Keep formatting clean and scannable
""",
        "twitter": f"""{base_prompt}
Twitter-Specific Elements:
- Crisp, concise messaging
- Natural voice with personality
- 2-3 relevant hashtags that flow naturally
- Conversation-starting element
- Memorable closing thought or hook
- Character-conscious structure
""",
    }
    return prompts.get(platform)


def get_platform_specific_image_prompt(platform, topic):
    """Modified to use Gemini for dynamic prompt generation"""
    try:
        ensure_api_keys_initialized()  # Ensure API keys are initialized

        prompt_generation_text = f"""
        Create a detailed prompt for generating an image that perfectly complements a {platform} post about {topic}.
        
        Requirements for the image:
        1. Must be visually engaging and platform-appropriate for {platform}
        2. Should capture the essence of the topic while being creative
        3. Include specific details about:
           - Style and artistic direction
           - Mood and atmosphere
           - Color palette
           - Composition elements
           - Key visual elements to include
        4. Must be safe for work and appropriate for all audiences
        
        Return only the image generation prompt, without any explanations or additional text.
        """

        model = app_config["gemini_model"]
        try:
            response = model.generate_content(
                prompt_generation_text,
                generation_config={"temperature": 0.7, "top_p": 0.8, "top_k": 40},
            )

            if not response or not response.text:
                raise Exception("Failed to generate image prompt")

            generated_prompt = response.text.strip()

            # Add platform-specific requirements
            if platform == "linkedin":
                return f"{generated_prompt}"
            else:  # twitter
                return f"{generated_prompt} Style: Bold, attention-grabbing, optimized for mobile viewing on Twitter."

        except Exception as e:
            print(f"Error generating prompt with Gemini: {str(e)}")
            raise e

    except Exception as e:
        print(f"Error in image prompt generation: {str(e)}")
        # Fallback to original static prompts if Gemini generation fails
        prompts = {
            "linkedin": f"""Create a image about {topic} for linkedin:
- Style: Clean, corporate, modern
- Headline: 6-8 words, short and impactful
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


@app.route("/")
def home():
    return "PostCraft API is live!"


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
            print("User registered successfully!" + new_user["name"]),
        )
    except Exception as e:
        print("Registration error:", str(e))
        return jsonify({"error": "Registration failed: " + str(e)}), 500


@app.route("/auth/login", methods=["POST"])
def login():
    try:
        data = request.json

        # Validate required fields
        if not data.get("email") or not data.get("password"):
            return jsonify({"error": "Email and password are required"}), 400

        # Find the user by email
        user = db.users.find_one({"email": data["email"]})

        # Validate credentials
        if not user or not check_password_hash(user["password"], data["password"]):
            return jsonify({"error": "Invalid email or password"}), 401

        # Fetch API keys from the user's database record
        hf_api_key = user.get("hf_api_key")
        gemini_api_key = user.get("gemini_api_key")

        # Initialize API keys if they exist
        if hf_api_key and gemini_api_key:
            try:
                # Configure Gemini API
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                # Store the initialized model and headers in app_config
                app_config["gemini_model"] = model
                app_config["gemini_key"] = gemini_api_key
                app_config["huggingface_key"] = hf_api_key
                app_config["hf_headers"] = {"Authorization": f"Bearer {hf_api_key}"}

                print("API keys successfully initialized for logged-in user!")
            except Exception as api_error:
                print(f"Failed to initialize API keys during login: {api_error}")
                # Optionally notify the user about the failure
                return (
                    jsonify(
                        {
                            "warning": "Failed to initialize API keys",
                            "details": str(api_error),
                        }
                    ),
                    200,
                )

        # Generate JWT token
        token = create_access_token(
            identity=str(user["_id"]), expires_delta=timedelta(days=7)
        )

        # Prepare the response
        response_data = {
            "token": token,
            "user": {
                "id": str(user["_id"]),
                "name": user["name"],
                "email": user["email"],
                "setup_completed": user.get("setup_completed", False),
            },
        }

        # Add API keys to the response if they exist
        if hf_api_key and gemini_api_key:
            response_data["api_keys"] = {
                "hf_api_key": hf_api_key,
                "gemini_api_key": gemini_api_key,
            }

        return (
            jsonify(response_data),
            200,
            print("User logged in successfully!" + user["name"]),
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

        # Log the API keys for debugging (be careful not to expose in production)
        print(
            f"Received API keys - HuggingFace: {hf_api_key}, Gemini: {gemini_api_key}"
        )

        # Validate API keys
        if not hf_api_key or not gemini_api_key:
            return jsonify({"error": "Both API keys are required"}), 400

        # Initialize Gemini API with robust error handling
        try:
            # Configure Gemini API
            genai.configure(api_key=gemini_api_key)

            # Create model with explicit error handling
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")

                # Verify model works with a test generation
                test_response = model.generate_content(
                    "Hello, can you confirm initialization?"
                )
                print("Gemini API test response:", test_response.text)

                # Store the actual model instance
                app_config["gemini_model"] = model
                app_config["gemini_key"] = gemini_api_key
                app_config["huggingface_key"] = hf_api_key
                app_config["hf_headers"] = {"Authorization": f"Bearer {hf_api_key}"}
                print("Gemini API successfully initialized and tested!")

            except Exception as model_error:
                print(f"Error creating or testing Gemini model: {model_error}")
                return (
                    jsonify(
                        {"error": f"Gemini model creation failed: {str(model_error)}"}
                    ),
                    500,
                )

        except Exception as api_error:
            print(f"Gemini API configuration error: {api_error}")
            return (
                jsonify(
                    {"error": f"Gemini API configuration failed: {str(api_error)}"}
                ),
                500,
            )

        # Save API keys to the user's database record
        user_id = get_jwt_identity()
        db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "hf_api_key": hf_api_key,
                    "gemini_api_key": gemini_api_key,
                    "setup_completed": True,
                }
            },
        )
        print("API keys saved to user record successfully!")
        return jsonify({"message": "APIs initialized successfully"}), 200

    except Exception as e:
        print(f"Initialization failed: {e}")
        return jsonify({"error": f"Initialization failed: {str(e)}"}), 500


def humanize_content(text, platform):
    """Post-process the generated content to make it more human-like"""
    try:
        # Remove common formulaic starts
        common_starts = [
            "okay, so",
            "well,",
            "you see,",
            "i've been thinking",
            "let me tell you",
        ]
        lower_text = text.lower()
        for start in common_starts:
            if lower_text.startswith(start):
                text = text[len(start) :].strip()
                text = text[0].upper() + text[1:]  # Capitalize first letter

        # Add natural variations and imperfections
        variations = {
            "definitely": ["def", "definitely", "for sure", "absolutely"],
            "amazing": ["amazing", "awesome", "fantastic", "great", "incredible"],
            "think": ["think", "believe", "feel", "sense"],
            "very": ["very", "really", "quite", "pretty"],
            "important": ["important", "crucial", "key", "essential"],
            "interesting": ["interesting", "fascinating", "intriguing", "compelling"],
        }

        # Replace some words with their variations randomly
        for word, alternatives in variations.items():
            if word in text.lower():
                text = text.replace(word, random.choice(alternatives))

        # Platform-specific humanization
        if platform == "twitter":
            # Make it more Twitter-like
            if random.random() < 0.3:  # 30% chance to use shorter forms
                text = text.replace("because", "bc")
                text = text.replace("with", "w/")
                text = text.replace("without", "w/o")

        # Clean up any artificial patterns
        text = text.replace("  ", " ").strip()

        return text
    except Exception as e:
        print(f"Error in humanize_content: {str(e)}")
        return text  # Return original text if processing fails


@app.route("/generate_post", methods=["POST"])
@jwt_required()
def generate_post():
    try:
        print("Generating post... start")
        ensure_api_keys_initialized()
        print("API keys initialized:", app_config.get("hf_headers") is not None)

        user_id = get_jwt_identity()
        data = request.json
        topic = data.get("topic")
        platform = data.get("platform")
        include_images = data.get("includeImages", True)
        image_count = data.get("imageCount", 1) if include_images else 0
        post_length = data.get("postLength", "medium")
        temperature = data.get("temperature", 0.7)

        if not topic or not platform:
            return jsonify({"error": "Topic and platform are required"}), 400

        # Generate text
        text_prompt = get_platform_specific_prompt(platform, topic, post_length)
        response = app_config["gemini_model"].generate_content(
            text_prompt,
            generation_config={"temperature": temperature, "top_p": 0.95, "top_k": 50},
        )
        best_content = humanize_content(
            response.text.strip().replace("**", "").replace("#", " #"), platform
        )

        # Generate images
        images = []
        image_error = None
        if include_images and app_config.get("hf_headers"):
            print("Attempting image generation...")
            for i in range(image_count):
                try:
                    print(f"Generating image {i + 1}/{image_count}...")
                    base_prompt = get_platform_specific_image_prompt(platform, topic)
                    image_response = requests.post(
                        app_config["hf_image_url"],
                        headers=app_config["hf_headers"],
                        json={
                            "inputs": base_prompt,
                            "options": {
                                "height": 256,
                                "width": 256,
                            },
                        },
                    )
                    if image_response.status_code == 200:
                        image_data = base64.b64encode(image_response.content).decode(
                            "utf-8"
                        )
                        images.append(f"data:image/jpeg;base64,{image_data}")
                    else:
                        print(
                            f"Image generation failed with status code {image_response.status_code}"
                        )
                        image_error = image_response.json().get("error")

                except requests.RequestException as req_err:
                    print(f"Image generation error: {req_err}")
                    images.append(None)  # Placeholder for failed image
        else:
            print("Image generation skipped: No HF headers or includeImages=False")

        print("Generating post image... end")
        # Calculate engagement score
        engagement_prompt = f"""
        Analyze this social media post for {platform} and provide an engagement score out of 100. Consider the following factors:

        - Readability: Is the post easy to read and understand?
        - Call to Action: Does it effectively encourage user interaction or response?
        - Emotional Appeal: Does the post resonate emotionally with the audience?
        - Hashtag Usage: Are hashtags used effectively and relevantly?
        - Length Appropriateness: Is the post the right length for the platform?
        {f'- Visual Impact: Does the post benefit from having {len(images)} image(s)?' if images else ''}

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
                "includes_images": include_images,
            },
        }

        db.users.update_one({"_id": ObjectId(user_id)}, {"$push": {"posts": post}})

        # Don't manually add CORS headers - let Flask-CORS handle it
        return (
            jsonify(
                {
                    "text": best_content,
                    "images": images,
                    "engagement_score": engagement_score,
                    "image_error": image_error if include_images else None,
                }
            ),
            200,
        )

    except Exception as e:
        print(f"Full error: {str(e)}")
        # Don't manually add CORS headers here either
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
        print("Posts fetched successfully!")
        return jsonify({"posts": posts}), 200

    except Exception as e:
        print(f"Error in get_user_posts: {str(e)}")
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
