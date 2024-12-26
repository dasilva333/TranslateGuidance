# Import the CustomSaveImage class from the custom_save_node.py file
from .translate_guidance_node import TranslateGuidanceNode

# Map the node name to the class
NODE_CLASS_MAPPINGS = {
    "Translate Flux Guidance": TranslateGuidanceNode,  # This is how it will appear in ComfyUI
}

# Optionally, map a display name to make it look more user-friendly
NODE_DISPLAY_NAME_MAPPINGS = {
    "Translate Flux Guidance": "🖼️ Translate Flux Guidance",
}
