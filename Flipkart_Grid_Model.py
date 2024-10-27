import re
import openpyxl
import cv2  # Import OpenCV for real-time capture
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import numpy as np
from google.colab import files

# Load the Qwen2 model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Create a new Excel workbook
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.title = "Product Analysis"

# Add headers
headers = ["Product Name", "Category", "Quantity", "Count", "Expiry Date", "Freshness Index", "Shelf Life"]
sheet.append(headers)

# Regular expression patterns to extract data
packaged_product_pattern = r"Product Name: (.*)\n  - Product Category: (.*)\n  - Product Quantity: (.*)\n  - Product Count: (.*)\n  - Expiry Date: (.*)"
fruits_vegetables_pattern = r"Type of fruit/vegetable: (.*)\n  - Freshness Index: (.*)\n  - Estimated Shelf Life: (.*)"


# Let user upload the image
uploaded = files.upload()

# Iterate over all uploaded files
for image_name in uploaded.keys():
    # Open the image using PIL
    image = Image.open(image_name)
    image = image.resize((512, 512))

    # Prepare the text prompt for predicting product details and freshness
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image_url": "Captured from webcam"  # Description for internal use
                },
                {
                    "type": "text",
                    "text": """This image contains fruits, vegetables, or packaged products.
                    Please analyze the image and provide:
                    - For packaged products:
                        - Product Name
                        - Product Category
                        - Product Quantity
                        - Product Count
                        - Expiry Date (if available)
                    - For fruits and vegetables:
                        - Type of fruit/vegetable
                        - Freshness Index (based on visual cues)
                        - Estimated Shelf Life"""
                }
            ]
        }
    ]

    # Prepare the text prompt for processing
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process the image and prompt for model input
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to GPU if available
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate output from the model
    output_ids = model.generate(**inputs, max_new_tokens=1024)

    # Decode the generated output
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    # Print the output text for the current image
    print(f"Analysis Output for the captured frame:")
    print(output_text)

    # Extract packaged product information
    packaged_product_match = re.search(packaged_product_pattern, output_text)
    fruits_vegetables_match = re.search(fruits_vegetables_pattern, output_text)

    if packaged_product_match:
        product_name = packaged_product_match.group(1).strip()
        category = packaged_product_match.group(2).strip()
        quantity = packaged_product_match.group(3).strip()
        count = packaged_product_match.group(4).strip()
        expiry_date = packaged_product_match.group(5).strip()
    else:
        product_name = category = quantity = count = expiry_date = "N/A"

    if fruits_vegetables_match:
        # Overwrite product fields with fruit/vegetable data if applicable
        product_name = fruits_vegetables_match.group(1).strip()
        category = "Fruit/Vegetable"
        freshness_index = fruits_vegetables_match.group(2).strip()
        shelf_life = fruits_vegetables_match.group(3).strip()
    else:
        freshness_index = shelf_life = "N/A"

    # Insert row into Excel
    sheet.append([product_name, category, quantity, count, expiry_date, freshness_index, shelf_life])

# Save the workbook
workbook.save("product_analysis.xlsx")
print("Data saved to product_analysis.xlsx")

