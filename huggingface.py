import customtkinter as ctk
import openai
from PIL import Image, ImageTk
import io
import requests
from datetime import datetime
import json
import os
from CTkMessagebox import CTkMessagebox
import google.generativeai as genai

# Set theme and color scheme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ModernPostGenerator:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Social Media Post Generator")
        self.window.geometry("1000x800")

        # Initialize APIs
        self.hf_headers = None
        self.hf_image_url = None
        self.gemini_model = None
        self.current_image = None

        # Configure grid layout
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=0)
        self.window.grid_rowconfigure(1, weight=1)

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        # Top Frame for API Configuration
        self.top_frame = ctk.CTkFrame(self.window, corner_radius=10)
        self.top_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # OpenAI API Key
        openai_label = ctk.CTkLabel(
            self.top_frame, text="OpenAI API Key:", font=("Helvetica", 12, "bold")
        )
        openai_label.pack(side="left", padx=10, pady=10)

        self.api_key_entry = ctk.CTkEntry(
            self.top_frame, width=300, placeholder_text="Enter your OpenAI API key"
        )
        self.api_key_entry.pack(side="left", padx=10, pady=10)

        # Gemini API Key
        gemini_label = ctk.CTkLabel(
            self.top_frame, text="Gemini API Key:", font=("Helvetica", 12, "bold")
        )
        gemini_label.pack(side="left", padx=10, pady=10)

        self.gemini_key_entry = ctk.CTkEntry(
            self.top_frame, width=300, placeholder_text="Enter your Gemini API key"
        )
        self.gemini_key_entry.pack(side="left", padx=10, pady=10)

        self.init_button = ctk.CTkButton(
            self.top_frame,
            text="Initialize API",
            command=self.initialize_apis,
            width=120,
        )
        self.init_button.pack(side="left", padx=10, pady=10)

        # Main Content Frame
        self.content_frame = ctk.CTkFrame(self.window, corner_radius=10)
        self.content_frame.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)

        # Left Column - Input and Text
        left_column = ctk.CTkFrame(self.content_frame, corner_radius=10)
        left_column.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Topic Input
        topic_label = ctk.CTkLabel(
            left_column, text="Topic/Context:", font=("Helvetica", 12, "bold")
        )
        topic_label.pack(padx=10, pady=(10, 5), anchor="w")

        self.topic_entry = ctk.CTkEntry(
            left_column, placeholder_text="Enter your topic here", height=35
        )
        self.topic_entry.pack(fill="x", padx=10, pady=(0, 10))

        # Platform Selection
        platform_label = ctk.CTkLabel(
            left_column, text="Platform:", font=("Helvetica", 12, "bold")
        )
        platform_label.pack(padx=10, pady=(10, 5), anchor="w")

        self.platform_var = ctk.StringVar(value="linkedin")
        platforms_frame = ctk.CTkFrame(left_column)
        platforms_frame.pack(fill="x", padx=10, pady=(0, 10))

        platforms = [
            ("LinkedIn", "linkedin"),
            ("Twitter", "twitter"),
            ("Instagram", "instagram"),
        ]
        for text, value in platforms:
            ctk.CTkRadioButton(
                platforms_frame, text=text, variable=self.platform_var, value=value
            ).pack(side="left", padx=10, pady=5)

        # Generate Button
        self.generate_button = ctk.CTkButton(
            left_column,
            text="Generate Post",
            command=self.generate_post,
            height=40,
            font=("Helvetica", 13, "bold"),
        )
        self.generate_button.pack(padx=10, pady=15)

        # Output Text
        self.output_text = ctk.CTkTextbox(left_column, wrap="word", height=300)
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Right Column - Image Display
        right_column = ctk.CTkFrame(self.content_frame, corner_radius=10)
        right_column.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        image_label = ctk.CTkLabel(
            right_column, text="Generated Image", font=("Helvetica", 12, "bold")
        )
        image_label.pack(padx=10, pady=10)

        self.image_frame = ctk.CTkFrame(right_column, corner_radius=10)
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(expand=True)

        # Save Button
        self.save_button = ctk.CTkButton(
            right_column,
            text="Save Post",
            command=self.save_post,
            height=40,
            font=("Helvetica", 13, "bold"),
        )
        self.save_button.pack(padx=10, pady=15)

    def load_config(self):
        try:
            with open("config_hf.json", "r") as f:
                config = json.load(f)
                self.api_key_entry.insert(0, config.get("huggingface_key", ""))
                self.gemini_key_entry.insert(0, config.get("gemini_key", ""))
                self.initialize_apis()
        except FileNotFoundError:
            pass

    def save_config(self):
        config = {
            "huggingface_key": self.api_key_entry.get(),
            "gemini_key": self.gemini_key_entry.get(),
        }
        with open("config_hf.json", "w") as f:
            json.dump(config, f)

    def initialize_apis(self):
        hf_key = self.api_key_entry.get()
        gemini_key = self.gemini_key_entry.get()

        try:
            if hf_key:
                self.hf_headers = {"Authorization": f"Bearer {hf_key}"}
                self.hf_image_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
                self.save_config()
                self.show_info("Success", "Hugging Face API initialized successfully")

            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")

            self.save_config()
            self.show_info("Success", "APIs initialized successfully")
        except Exception as e:
            self.show_error(f"Failed to initialize APIs: {str(e)}")

    def generate_image(self, prompt):
        try:
            # Make the request to Hugging Face's image generation model
            response = requests.post(
                self.hf_image_url,
                headers=self.hf_headers,
                json={"inputs": prompt},
            )

            # Check if the request was successful
            if response.status_code != 200:
                raise Exception(
                    f"Hugging Face API Error: {response.status_code} {response.text}"
                )

            # Load the image from the response
            image_data = Image.open(io.BytesIO(response.content))

            # Resize image to fit in the window
            width = 400
            ratio = width / image_data.width
            height = int(image_data.height * ratio)
            image_data = image_data.resize((width, height))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image_data)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            self.current_image = image_data
            return True

        except Exception as e:
            self.show_error(f"Failed to generate image: {str(e)}")
            return None

    def get_platform_specific_prompt(self, platform, topic):
        prompts = {
            "linkedin": f"""Create a single professional LinkedIn post about {topic}.
Requirements:
- Professional and thoughtful tone
- 4-5 short paragraphs maximum
- Include a thought-provoking question to encourage engagement
- Add relevant professional hashtags at end of post
- Keep focus on business/professional implications
- Do not mention other social media platforms
- Maximum 1400 characters""",
            "twitter": f"""Create a single impactful Twitter post about {topic}.
Requirements:
- Concise and direct message
- Maximum 280 characters including hashtags
- Include 2-3 relevant hashtags
- Clear call to action
- Professional, playful and engaging tone
- Do not mention other social media platforms""",
            "instagram": f"""Create a single engaging Instagram post about {topic}.
Requirements:
- Visually descriptive language
- Playful and engaging tone
- Emoji usage welcome but not excessive
- 3-4 relevant hashtags at the end of the post
- Clear call to action
- Do not mention other social media platforms
- No 'swipe' or 'carousel' mentions
- Focus on a two powerful but playful message""",
        }
        return prompts.get(platform)

    def get_platform_specific_image_prompt(self, platform, topic):
        prompts = {
            "linkedin": f"""Create a professional marketing image for LinkedIn about {topic}.
Requirements:
- Clean, corporate style
- Professional color scheme (blues, grays)
- Maximum one short text headline
- Minimum text, maximum impact
- Business-appropriate imagery
- High contrast for readability""",
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

    def generate_post(self):
        if not self.hf_headers or not self.hf_image_url or not self.gemini_model:
            self.show_error("Please initialize APIs first")
            return

        topic = self.topic_entry.get().strip()
        if not topic:
            self.show_warning("Please enter a topic")
            return

        platform = self.platform_var.get()

        try:
            # Get platform-specific prompts
            text_prompt = self.get_platform_specific_prompt(platform, topic)
            image_prompt = self.get_platform_specific_image_prompt(platform, topic)

            # Generate content
            response = self.gemini_model.generate_content(text_prompt)
            post_content = response.text.strip()

            # Clean up the content
            post_content = post_content.replace("**", "").replace("#", " #")
            if platform == "twitter" and len(post_content) > 280:
                post_content = post_content[:277] + "..."

            # Generate image
            image_url = self.generate_image(image_prompt)

            self.output_text.delete("0.0", "end")
            self.output_text.insert("0.0", post_content)

            self.current_post = {
                "platform": platform,
                "topic": topic,
                "content": post_content,
                "image_url": image_url,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }

        except Exception as e:
            self.show_error(f"Failed to generate post: {str(e)}")

    def save_post(self):
        if not hasattr(self, "current_post"):
            self.show_warning("Please generate a post first")
            return

        try:
            os.makedirs("posts", exist_ok=True)
            filename = f"posts/{self.current_post['timestamp']}_{self.current_post['platform']}"

            # Save post data as JSON
            json_filename = f"{filename}.json"
            with open(json_filename, "w") as f:
                json.dump(self.current_post, f, indent=4)

            # Save image if available
            if hasattr(self, "current_image") and self.current_image:
                image_filename = f"{filename}.png"
                self.current_image.save(image_filename)

            self.show_info("Success", f"Post saved to {json_filename}")

        except Exception as e:
            self.show_error(f"Failed to save post: {str(e)}")

    def show_error(self, message):
        CTkMessagebox(title="Error", message=message, icon="cancel")

    def show_warning(self, message):
        CTkMessagebox(title="Warning", message=message, icon="warning")

    def show_info(self, title, message):
        CTkMessagebox(title=title, message=message, icon="check")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = ModernPostGenerator()
    app.run()
