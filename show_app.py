import gradio as gr
import replicate
import requests
from PIL import Image
import io
import os
import logging
from dotenv import load_dotenv
import uuid

load_dotenv()

# Set up logging
logging.basicConfig(
    filename='system_usage.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

#########################################################
#LLAMADA A FLUX PULL-ID EN REPLICATE
#########################################################

def generate_image_pulid_flux(face_image, prompt, width, height, neg_prompt, quality, seed, face_refinement_steps):
    # Create a unique temporary file name
    import uuid
    temp_image_path = f"temp_image_{uuid.uuid4()}.png"
    
    try:
        # Save the uploaded image temporarily
        face_image.save(temp_image_path)
        
        # Convert seed to integer
        try:
            seed_value = int(seed)
        except ValueError:
            seed_value = -1  # Default to -1 if conversion fails

        output = replicate.run(
            "zsxkib/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
            input={
                "prompt": prompt, 
                "width": width,
                "height": height,
                "true_cfg": 1,
                "id_weight": 1, # 
                "num_steps": 20, 
                "start_step": 1,
                "num_outputs": 1,
                "output_format": "png",
                "guidance_scale": 4,
                "output_quality": 100,
                "main_face_image": open(temp_image_path, "rb"),
                "negative_prompt": neg_prompt,
                "max_sequence_length": 128,
                "seed": seed_value,  # Use the converted integer value
                "timestep_to_start_cfg": 1
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
    finally:
        # Close any open file handles before trying to remove
        try:
            if 'output' in locals() and hasattr(output, 'close'):
                output.close()
        except:
            pass
            
        # Try to remove the temporary file
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")
            
#########################################################
#LLAMADA A STORYFACE
#########################################################

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

#########################################################
#iTERACION DE N LLAMADAS A STORYFACE
#########################################################
    
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

#########################################################
#PIPELINE DE LAS 3 FUNCIONES ANTERIORES
#########################################################

def process_all(face_image, prompt, width, height, neg_prompt, quality, seed, face_refinement_steps):
    logger.info("System used")
    
    if face_image is None:
        return None, None

    # Generate initial image with PuLID-FLUX
    pulid_flux_result = generate_image_pulid_flux(
        face_image, prompt, width, height, neg_prompt, quality, seed, face_refinement_steps
    )
    
    if pulid_flux_result is None:
        return None, None

    # Apply iterative face swap
    storyface_result = iterative_face_swap(face_image, pulid_flux_result, face_refinement_steps, quality)
    return pulid_flux_result, storyface_result

#########################################################
#GRADIO WEBAPP
#########################################################

with gr.Blocks(title="Natasquad Image Generation Playground") as demo:
    gr.Markdown("# Natasquad Image Generation Playground")
    
    with gr.Row():
        with gr.Column():
            
            #########################################################
            #PARAMETROS DE LLAMADA A REPLICATE
            #########################################################            
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
            
            submit_button = gr.Button("Generate Images")
        
        with gr.Column():            
            output_pulid_flux = gr.Image(label="Initial Generation")
            output_storyface = gr.Image(label="Result after Face Refinement")
               
    submit_button.click(
        process_all,
        inputs=[
            face_image, prompt, width, height, neg_prompt, quality, seed, face_refinement_steps
        ],
        outputs=[output_pulid_flux, output_storyface]
    )

if __name__ == "__main__":
    demo.show_api = False
    demo.launch(server_port=7860)