import cv2, numpy as np, torch, trimesh
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from rembg import remove
from tsr.system import TS
device = "cuda"

# --- load once at startup ---
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16)
sd = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    torch_dtype=torch.float16).to(device)
sd.scheduler = UniPCMultistepScheduler.from_config(sd.scheduler.config)

tsr = TSR.from_pretrained("stabilityai/TripoSR",
        config_name="config.yaml", weight_name="model.ckpt").to(device)
tsr.renderer.set_chunk_size(8192)


def preprocess(img: Image.Image) -> Image.Image:
    g = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (512, 512))
    # scribble ControlNet wants WHITE lines on BLACK background
    if g.mean() > 127:           # drawn as black-on-white -> invert
        g = 255 - g
    _, g = cv2.threshold(g, 40, 255, cv2.THRESH_BINARY)
    g = cv2.dilate(g, np.ones((3, 3), np.uint8))   # thicken thin strokes
    return Image.fromarray(g).convert("RGB")

def sketch_to_3d(sketch, prompt=""):
    cond = preprocess(sketch)
    full_prompt = f"{prompt}, a single object, product render, centered, " \
                  "plain white background, studio lighting, high detail"
    img = sd(full_prompt, image=cond, num_inference_steps=25,
             controlnet_conditioning_scale=0.9,
             negative_prompt="blurry, multiple objects, text, low quality").images[0]
    fg = remove(img)                                  # RGBA
    bg = Image.new("RGBA", fg.size, (127, 127, 127, 255))   # TripoSR likes gray bg
    obj = Image.alpha_composite(bg, fg).convert("RGB")
    with torch.no_grad():
        codes = tsr([obj], device=device)
    mesh = tsr.extract_mesh(codes, True, resolution=256)[0]
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0, "Empty mesh"
    mesh.export("output.glb")
    trimesh.load("output.glb")                        # reopen check
    return img, "output.glb"
