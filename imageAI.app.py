# Import dependencies
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk
from authtoken import auth_token, folder_path # authtoken and folder_path are unique to your API Keys and local directories
import torch
from diffusers import StableDiffusionPipeline
from tomesd import apply_patch
import xformers
import threading
from accelerate import Accelerator
import json
import os

# Specify the folder path (same as in the summarization script)
folder_path = folder_path

# Load key points from JSON file
json_file_path = os.path.join(folder_path, 'descriptions.json')
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    descriptions_dict = json.load(json_file)

# Initialize app
app = tk.Tk()
app.geometry("532x622")
app.title('Stable Diffusion Test')
ctk.set_appearance_mode("dark")

# Dropdown menu initialization
selected_file = tk.StringVar()
file_dropdown = ttk.Combobox(app, textvariable=selected_file, values=list(descriptions_dict.keys()), height=40, font=("Arial", 12))
file_dropdown.place(x=10, y=10)

# Image configuration
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.configure(text="")
lmain.place(x=10, y=160)

# Progress bar initialization
progress = ttk.Progressbar(app, orient='horizontal', length=512, mode='determinate')
progress.place(x=10, y=110)

# Setup accelerator
accelerator = Accelerator()

# Load model without torch_dtype=torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()
pipe.to(accelerator.device)

apply_patch(pipe.unet)

def generate():
    filename = selected_file.get()
    if filename not in descriptions_dict:
        return
    descriptions = descriptions_dict[filename]
    prompt_text = f"An image representing the provided description: {descriptions}"

    def run_generation():
        num_inference_steps = 25
        guidance_scale = 8.5
        intermediate_images = []

        def callback(step, timestep, latents):
            # Update progress bar
            progress['value'] = (step / num_inference_steps) * 100
            app.update_idletasks()

            # Decode latents to an image and update GUI with intermediate image
            with torch.no_grad():
                images = pipe.decode_latents(latents)
            image = pipe.numpy_to_pil(images)[0]
            intermediate_images.append(image)
            img = ImageTk.PhotoImage(image)
            lmain.configure(image=img)
            lmain.image = img

        # Generate the image with intermediate steps and callback
        result_image = pipe(prompt_text, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, callback=callback, callback_steps=1).images[0]
        save_path = os.path.join(folder_path, f"{filename}_generated.png")
        result_image.save(save_path)

    threading.Thread(target=run_generation).start()

# Button configuration
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
