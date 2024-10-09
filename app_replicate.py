import gradio as gr
import replicate
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

# Make sure to set your REPLICATE_API_TOKEN in your environment variables
# os.environ["REPLICATE_API_TOKEN"] = "your-token-here"

def generate_image_pulid_flux(prompt, id_image, width, height, num_steps, neg_prompt, max_sequence_length):
    # Save the uploaded image temporarily
    temp_image_path = "temp_image.png"
    id_image.save(temp_image_path)

    try:
        output = replicate.run(
            "zsxkib/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
            input={
                "width": width,
                "height": height,
                "prompt": prompt,
                "true_cfg": 1,
                "id_weight": 1,
                "num_steps": num_steps,
                "start_step": 0,
                "num_outputs": 1,
                "output_format": "png",
                "guidance_scale": 4,
                "output_quality": 100,
                "main_face_image": open(temp_image_path, "rb"),
                "negative_prompt": neg_prompt,
                "max_sequence_length": max_sequence_length
            }
        )
        
        # The output is expected to be a list with one URL
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
        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

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
            prompt = gr.Textbox(
                label="Prompt", 
                value="viking, realism, fantasy, dark",
            )            
            width = gr.Slider(256, 1536, 896, step=16, label="Width")
            height = gr.Slider(256, 1536, 1152, step=16, label="Height")            
            num_steps = gr.Slider(1, 20, 20, step=1, label="Number of steps")            
            max_sequence_length = gr.Slider(
                128, 512, 128, step=128, 
                label="Max Sequence Length",                
            )            
            neg_prompt = gr.Textbox(
                label="Negative Prompt",
                value="bad quality, worst quality, text, signature, watermark, extra limbs",               
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
    demo.launch(server_name="0.0.0.0", server_port=7860)