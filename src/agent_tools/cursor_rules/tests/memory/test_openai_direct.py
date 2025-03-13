from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()

# Get the absolute path to the image
project_root = Path(__file__).parent.parent.parent
image_path = project_root / "images" / "spartan.png"
assert image_path.exists(), f"Image not found at {image_path}"

# Read and encode the image
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                },
            },
        ],
    }],
)

print(response.choices[0].message.content) 