import gradio as gr
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from PIL import Image
import io
import os
import base64
import time
import json
from dotenv import load_dotenv

load_dotenv()

def create_session_with_retries():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def generate_image_pulid_flux(prompt, id_image, width, height, num_steps, neg_prompt, max_sequence_length):
    # Convert PIL Image to base64
    img_byte_arr = io.BytesIO()
    id_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

    # Prepare the payload
    payload = {
        "data": [
            prompt,
            base64_image,
            0,  # start_step
            4,  # guidance
            "-1",  # seed
            1,  # true_cfg
            width,
            height,
            num_steps,
            1,  # id_weight
            neg_prompt,
            1,  # timestep_to_start_cfg
            max_sequence_length
        ]
    }

    # Calculate timeout based on number of steps (adjust this formula as needed)
    timeout = max(300, num_steps * 10)  # Minimum 5 minutes, then 10 seconds per step

    session = create_session_with_retries()

    try:
        print(f"Starting image generation with a timeout of {timeout} seconds...")
        # Initiate the job
        response = session.post(
            "https://yanze-pulid-flux.hf.space/call/generate_image",
            json=payload,
            timeout=30  # Short timeout for initial request
        )
        response.raise_for_status()
        event_id = response.json()

        # Poll for results
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result_response = session.get(f"https://yanze-pulid-flux.hf.space/call/generate_image/{event_id}")
                result_response.raise_for_status()
                result_data = result_response.json()
                
                if 'error' in result_data:
                    raise Exception(f"API Error: {result_data['error']}")
                
                if result_data.get('status') == 'COMPLETE':
                    print("Image generation completed successfully.")
                    # Assuming the image is returned as a base64 string
                    image_data = base64.b64decode(result_data['data'][0])
                    return Image.open(io.BytesIO(image_data))
                
                time.sleep(5)  # Wait for 5 seconds before polling again
            except json.JSONDecodeError:
                print("Received incomplete response. Retrying...")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"Error while polling: {str(e)}. Retrying...")
                time.sleep(1)

        raise TimeoutError("The operation timed out")

    except requests.exceptions.RequestException as e:
        print(f"Error in PuLID-FLUX generation: {str(e)}")
    except TimeoutError as e:
        print(str(e))
    except Exception as e:
        print(f"Unexpected error in PuLID-FLUX generation: {str(e)}")
    
    return None

def process_images_storyface(face_image, model_image, quality=100):
    face_img_bytes = io.BytesIO()
    face_image.save(face_img_bytes, format='PNG')
    model_img_bytes = io.BytesIO()
    model_image.save(model_img_bytes, format='PNG')

    url = os.getenv('URL')
    if not url:
        print("Error: StoryFace API URL not found in environment variables.")
        return None

    files = [
        ('images', ('face.png', face_img_bytes.getvalue(), 'image/png')),
        ('images', ('model.png', model_img_bytes.getvalue(), 'image/png'))
    ]
    data = {
        'watermark': 0,
        'quality': quality
    }

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Error in StoryFace processing: {str(e)}")
        return None

def process_all(face_image, prompt, width, height, num_steps, neg_prompt, max_sequence_length, quality):
    if face_image is None:
        return None, None

    pulid_flux_result = generate_image_pulid_flux(prompt, face_image, width, height, num_steps, neg_prompt, max_sequence_length)
    if pulid_flux_result is None:
        return None, None

    storyface_result = process_images_storyface(face_image, pulid_flux_result, quality)
    return pulid_flux_result, storyface_result

with gr.Blocks(title="Natasquad Image Generation Playground") as demo:
    gr.Markdown("# Natasquad Image Generation Playground")
    
    with gr.Row():
        with gr.Column():
            face_image = gr.Image(label="Face Image", type="pil")
            prompt = gr.Textbox(label="Prompt", value="portrait, color, cinematic")
            width = gr.Slider(256, 1536, 896, step=16, label="Width")
            height = gr.Slider(256, 1536, 1152, step=16, label="Height")
            num_steps = gr.Slider(1, 20, 20, step=1, label="Number of steps")
            max_sequence_length = gr.Slider(128, 512, 128, step=128, label="Max Sequence Length")
            neg_prompt = gr.Textbox(
                label="Negative Prompt",
                value="bad quality, worst quality, text, signature, watermark, extra limbs"
            )
            quality = gr.Slider(1, 100, 100, step=1, label="Quality")
            submit_button = gr.Button("Generate Images")
        
        with gr.Column():            
            output_pulid_flux = gr.Image(label="Generated Image")
            output_storyface = gr.Image(label="Face Swap Refinement")

    submit_button.click(
        process_all,
        inputs=[face_image, prompt, width, height, num_steps, neg_prompt, max_sequence_length, quality],
        outputs=[output_pulid_flux, output_storyface]
    )

if __name__ == "__main__":
    demo.launch()