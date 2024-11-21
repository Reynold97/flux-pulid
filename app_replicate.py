import gradio as gr
import replicate
import requests
from PIL import Image
import io
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    filename='system_usage.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_image_pulid_flux(prompt, id_image, width, height, num_steps, neg_prompt, max_sequence_length,
                            id_weight, start_step, guidance_scale, seed, true_cfg, timestep_to_start_cfg):
    try:
        # Convert PIL Image to base64
        img_byte_arr = io.BytesIO()
        id_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        output = replicate.run(
            "zsxkib/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "true_cfg": true_cfg,
                "id_weight": id_weight,
                "num_steps": num_steps,
                "start_step": start_step,
                "num_outputs": 1,
                "output_format": "png",
                "guidance_scale": guidance_scale,
                "output_quality": 100,
                "main_face_image": base64_image,  # Send base64 encoded image
                "negative_prompt": neg_prompt,
                "max_sequence_length": max_sequence_length,
                "seed": seed,
                "timestep_to_start_cfg": timestep_to_start_cfg
            }
        )
        
        if output and isinstance(output, list) and len(output) > 0:
            image_url = output[0]
            response = requests.get(image_url)
            return Image.open(io.BytesIO(response.content))
        else:
            print("Unexpected output format from Replicate API")
            return None

    except Exception as e:
        print(f"Error in PuLID-FLUX generation: {str(e)}")
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
    
def iterative_face_swap(face_image, initial_model_image, refinement_steps, quality=100):
    """
    Iteratively apply face swap, using each result as the new model image.
    Always uses the original face_image as the source face.
    """
    current_model = initial_model_image
    
    for step in range(refinement_steps):
        result = process_images_storyface(face_image, current_model, quality)
        if result is None:
            return current_model  # Return last successful result
        current_model = result
    
    return current_model

def process_all(face_image, prompt, width, height, num_steps, neg_prompt, max_sequence_length, quality,
                id_weight, start_step, guidance_scale, seed, true_cfg, timestep_to_start_cfg, face_refinement_steps):
    logger.info("System used")
    
    if face_image is None:
        return None, None

    # Generate initial image with PuLID-FLUX
    pulid_flux_result = generate_image_pulid_flux(
        prompt, face_image, width, height, num_steps, neg_prompt, max_sequence_length,
        id_weight, start_step, guidance_scale, seed, true_cfg, timestep_to_start_cfg
    )
    
    if pulid_flux_result is None:
        return None, None

    # Apply iterative face swap
    storyface_result = iterative_face_swap(face_image, pulid_flux_result, face_refinement_steps, quality)
    return pulid_flux_result, storyface_result

with gr.Blocks(title="Natasquad Image Generation Playground") as demo:
    gr.Markdown("# Natasquad Image Generation Playground")
    
    with gr.Row():
        with gr.Column():
            # Basic Parameters
            gr.Markdown("### Basic Parameters")
            face_image = gr.Image(label="Face Image - Upload a clear image of a face", type="pil")
            prompt = gr.Textbox(
                label="Prompt - Describe the image you want to generate",
                value="portrait, color, cinematic"
            )
            neg_prompt = gr.Textbox(
                    label="Negative Prompt - Describe what you don't want in the image",
                    value="bad quality, worst quality, text, signature, watermark, extra limbs"
            )
            face_refinement_steps = gr.Slider(
                minimum=1, maximum=5, value=1, step=1,
                label="Face Refinement Steps - Higher values may give better results but take longer"
            )
            quality = gr.Slider(
                minimum=1, maximum=100, value=100, step=1,
                label="Face Refinement Quality"
            )
            seed = gr.Textbox(
                    value="-1",
                    label="Seed (-1 for random) - Set a specific seed for reproducible results"
            )
            
            # Image Size Controls
            gr.Markdown("### Image Size")
            width = gr.Slider(minimum=256, maximum=1536, value=896, step=16, label="Width")
            height = gr.Slider(minimum=256, maximum=1536, value=1152, step=16, label="Height")
            
            # Generation Parameters
            gr.Markdown("### Generation Parameters")
            id_weight = gr.Slider(
                minimum=0.0, maximum=3.0, value=1, step=0.05,
                label="ID Weight - Controls how much the generated image resembles the input face"
            )
            num_steps = gr.Slider(
                minimum=1, maximum=20, value=20, step=1,
                label="Number of Steps - More steps generally result in better quality"
            )
            start_step = gr.Slider(
                minimum=0, maximum=10, value=0, step=1,
                label="Start Step - Timestep to start inserting ID"
            )
            guidance_scale = gr.Slider(
                minimum=1.0, maximum=10.0, value=4, step=0.1,
                label="Guidance Scale - Controls how closely the image follows the prompt"
            )
                
            # Advanced Parameters
            with gr.Accordion("Advanced Options", open=False):                
                max_sequence_length = gr.Slider(
                    minimum=128, maximum=512, value=128, step=128,
                    label="Max Sequence Length - Longer sequences allow for more detailed prompts but may be slower"
                )               
                true_cfg = gr.Slider(
                    minimum=1.0, maximum=10.0, value=1, step=0.1,
                    label="True CFG Scale - Advanced CFG control (>1 means use true CFG)"
                )
                timestep_to_start_cfg = gr.Slider(
                    minimum=0, maximum=20, value=1, step=1,
                    label="Timestep to Start CFG"
                )             
            
            
            submit_button = gr.Button("Generate Images")
        
        with gr.Column():            
            output_pulid_flux = gr.Image(label="Generated Image by PuLID-FLUX")
            output_storyface = gr.Image(label="Final Result after Face Swap")

    submit_button.click(
        process_all,
        inputs=[
            face_image, prompt, width, height, num_steps, neg_prompt, max_sequence_length, quality,
            id_weight, start_step, guidance_scale, seed, true_cfg, timestep_to_start_cfg, face_refinement_steps
        ],
        outputs=[output_pulid_flux, output_storyface]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)