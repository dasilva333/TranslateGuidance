from datetime import datetime
from torch import Tensor
import torch
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .translate_guidance_lib import translate_guidance, FORMULAS, PATTERNS
from comfy.ldm.flux.layers import (
    timestep_embedding,
)
log_file = os.path.join(os.path.dirname(__file__), "guidance_output.csv")

# Global flag for negative vs. positive
is_negative_conditioning = True

def custom_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
) -> Tensor:
    # Access the global flag (if declared at module level)
    global is_negative_conditioning

    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # 1) Input transformations
    #    (identical to ComfyUI’s original block)
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

    # 2) Guidance embedding – extended with negative vs. positive logic
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance-distilled model.")

        if is_negative_conditioning:
            # Negative guidance config
            current_formula = transformer_options.get("negative_guidance_formula", None)
            current_pattern = transformer_options.get("negative_pattern", None)
            lowest_g = transformer_options.get("negative_lowest_guidance", 0.0)
            median_g = transformer_options.get("negative_median_guidance", 1.0)
            highest_g = transformer_options.get("negative_highest_guidance", 2.0)
            formula_scale = transformer_options.get("negative_guidance_scale", 1.0)
            pattern_scale = transformer_options.get("negative_pattern_scale", 1.0)
        else:
            # Positive guidance config
            current_formula = transformer_options.get("positive_guidance_formula", None)
            current_pattern = transformer_options.get("positive_pattern", None)
            lowest_g = transformer_options.get("positive_lowest_guidance", 1.0)
            median_g = transformer_options.get("positive_median_guidance", 3.5)
            highest_g = transformer_options.get("positive_highest_guidance", 6.0)
            formula_scale = transformer_options.get("positive_guidance_scale", 1.0)
            pattern_scale = transformer_options.get("positive_pattern_scale", 1.0)

        # print(f"Current formula: {current_formula}")
        if current_formula and current_formula != "None":
            translate_config = {
                "formula_name": current_formula,
                "pattern_name": current_pattern,
                "min_guidance": lowest_g,
                "mid_guidance": median_g,
                "max_guidance": highest_g,
                "formula_scale": formula_scale,
                "pattern_scale": pattern_scale,
                "is_negative": is_negative_conditioning,
            }
            translated_guidance = translate_guidance(
                timestep=timesteps,
                guidance=guidance,
                device=img.device,
                config=translate_config
            )
            # print(f"Translated guidance: {translated_guidance}", translate_config)
            output_guidance = translated_guidance
            vec = vec + self.guidance_in(
                timestep_embedding(translated_guidance, 256).to(img.dtype)
            )
        else:
            # Default behavior
            output_guidance = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(output_guidance.to(img.dtype))

        # Log the guidance usage
        log_guidance_values(
            guidance_method=current_formula,
            input_guidance=guidance,
            output_guidance=output_guidance.mean().item(),
            is_negative_conditioning=is_negative_conditioning
        )

    # 3) Combine other embeddings
    #    (original)
    vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
    txt = self.txt_in(txt)

    # 4) Prepare positional embeddings
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    # 5) Double-stream blocks (unchanged, except for optional patch replacing)
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"]
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": img, "txt": txt, "vec": vec, "pe": pe},
                {"original_block": block_wrap}
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        if control is not None:  # ControlNet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    # Merge text + image streams
    img = torch.cat((txt, img), 1)

    # 6) Single-stream blocks (same patch logic)
    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"])
                return out

            out = blocks_replace[("single_block", i)](
                {"img": img, "vec": vec, "pe": pe},
                {"original_block": block_wrap}
            )
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe)

        if control is not None:  # ControlNet outputs
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    # Extract out the image portion
    img = img[:, txt.shape[1]:, ...]

    # 7) Final layer (unchanged)
    img = self.final_layer(img, vec)

    # Toggle the global negative vs. positive condition for the next pass
    is_negative_conditioning = not is_negative_conditioning

    return img

def log_guidance_values(guidance_method, input_guidance, output_guidance, is_negative_conditioning):
    if torch.is_tensor(input_guidance):
        input_guidance = input_guidance.item()
    if torch.is_tensor(output_guidance):
        output_guidance = output_guidance.item()

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            guidance_method if guidance_method else "None",
            input_guidance,
            output_guidance,
            is_negative_conditioning
        ])

class BaseTranslateGuidanceNode:
    def initialize_log_file(self):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Guidance Method',
                'Input Guidance',
                'Output Guidance',
                'Negative Conditioning'
            ])

class TranslateGuidanceSimpleNode(BaseTranslateGuidanceNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_method": (FORMULAS,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "simple/model"

    def main(self, model, guidance_method):
        self.initialize_log_file()
        m = model.clone()

        # Do this to properly bind the function as a method:
        m.model.diffusion_model.forward_orig = custom_forward_orig.__get__(
            m.model.diffusion_model, 
            m.model.diffusion_model.__class__
        )

        if not hasattr(m, "model_options") or not isinstance(m.model_options, dict):
            m.model_options = {}
        if "transformer_options" not in m.model_options:
            m.model_options["transformer_options"] = {}

        # Simple usage: set both positive and negative to the same formula
        if guidance_method is not None and guidance_method != "None":
            m.model_options["transformer_options"].update({
                "positive_guidance_formula": guidance_method,
                "negative_guidance_formula": guidance_method
            })
        else:
            m.model_options["transformer_options"].pop("positive_guidance_formula", None)
            m.model_options["transformer_options"].pop("negative_guidance_formula", None)

        return (m,)

class TranslateGuidanceStandardNode(BaseTranslateGuidanceNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_guidance_method": (FORMULAS,),
                "negative_guidance_method": (FORMULAS,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "standard/model"

    def main(self, model, positive_guidance_method, negative_guidance_method):
        self.initialize_log_file()
        m = model.clone()

        # Do this to properly bind the function as a method:
        m.model.diffusion_model.forward_orig = custom_forward_orig.__get__(
            m.model.diffusion_model, 
            m.model.diffusion_model.__class__
        )

        if not hasattr(m, "model_options") or not isinstance(m.model_options, dict):
            m.model_options = {}
        if "transformer_options" not in m.model_options:
            m.model_options["transformer_options"] = {}

        # Default values are stored but formula can be overridden
        if positive_guidance_method is not None and positive_guidance_method != "None":
            m.model_options["transformer_options"]["positive_guidance_formula"] = positive_guidance_method
        else:
            m.model_options["transformer_options"].pop("positive_guidance_formula", None)

        if negative_guidance_method is not None and negative_guidance_method != "None":
            m.model_options["transformer_options"]["negative_guidance_formula"] = negative_guidance_method
        else:
            m.model_options["transformer_options"].pop("negative_guidance_formula", None)

        return (m,)

class TranslateGuidanceAdvancedNode(BaseTranslateGuidanceNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),

                "positive_guidance_formula": (FORMULAS, {"default": "None"}),
                "positive_pattern": (PATTERNS, {"default": "None"}),
                "positive_lowest_guidance": ("FLOAT", {"default": 0.5, "min": 0.0}),
                "positive_median_guidance": ("FLOAT", {"default": 3.5, "min": 0.0}),
                "positive_highest_guidance": ("FLOAT", {"default": 6.0, "min": 0.0}),
                "positive_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "positive_pattern_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),

                "negative_guidance_formula": (FORMULAS, {"default": "None"}),
                "negative_pattern": (PATTERNS, {"default": "None"}),
                "negative_lowest_guidance": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "negative_median_guidance": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "negative_highest_guidance": ("FLOAT", {"default": 2.0, "min": 0.0}),
                "negative_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "negative_pattern_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "advanced/model"

    def main(self,
             model,
             positive_guidance_formula,
             positive_pattern,
             positive_lowest_guidance,
             positive_median_guidance,
             positive_highest_guidance,
             positive_guidance_scale,
             positive_pattern_scale,
             negative_guidance_formula,
             negative_pattern,
             negative_lowest_guidance,
             negative_median_guidance,
             negative_highest_guidance,
             negative_guidance_scale,
             negative_pattern_scale
    ):
        self.initialize_log_file()
        m = model.clone()

        # Do this to properly bind the function as a method:
        m.model.diffusion_model.forward_orig = custom_forward_orig.__get__(
            m.model.diffusion_model, 
            m.model.diffusion_model.__class__
        )

        if not hasattr(m, "model_options") or not isinstance(m.model_options, dict):
            m.model_options = {}
        if "transformer_options" not in m.model_options:
            m.model_options["transformer_options"] = {}

        # Positive
        if positive_guidance_formula != "None":
            m.model_options["transformer_options"].update({
                "positive_guidance_formula": positive_guidance_formula,
                "positive_pattern": positive_pattern,
                "positive_lowest_guidance": positive_lowest_guidance,
                "positive_median_guidance": positive_median_guidance,
                "positive_highest_guidance": positive_highest_guidance,
                "positive_guidance_scale": positive_guidance_scale,
                "positive_pattern_scale": positive_pattern_scale,
            })
        else:
            m.model_options["transformer_options"].pop("positive_guidance_formula", None)
            m.model_options["transformer_options"].pop("positive_pattern", None)
            m.model_options["transformer_options"].pop("positive_lowest_guidance", None)
            m.model_options["transformer_options"].pop("positive_median_guidance", None)
            m.model_options["transformer_options"].pop("positive_highest_guidance", None)
            m.model_options["transformer_options"].pop("positive_guidance_scale", None)
            m.model_options["transformer_options"].pop("positive_pattern_scale", None)

        # Negative
        if negative_guidance_formula != "None":
            m.model_options["transformer_options"].update({
                "negative_guidance_formula": negative_guidance_formula,
                "negative_pattern": negative_pattern,
                "negative_lowest_guidance": negative_lowest_guidance,
                "negative_median_guidance": negative_median_guidance,
                "negative_highest_guidance": negative_highest_guidance,
                "negative_guidance_scale": negative_guidance_scale,
                "negative_pattern_scale": negative_pattern_scale,
            })
        else:
            m.model_options["transformer_options"].pop("negative_guidance_formula", None)
            m.model_options["transformer_options"].pop("negative_pattern", None)
            m.model_options["transformer_options"].pop("negative_lowest_guidance", None)
            m.model_options["transformer_options"].pop("negative_median_guidance", None)
            m.model_options["transformer_options"].pop("negative_highest_guidance", None)
            m.model_options["transformer_options"].pop("negative_guidance_scale", None)
            m.model_options["transformer_options"].pop("negative_pattern_scale", None)

        return (m,)

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False
        
any_type = AlwaysEqualProxy("*")

class TranslateGuidanceChartNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"anything": (any_type, {})}, "optional": {},
                }

    
    FUNCTION = "main"
    # OUTPUT_NODE = True    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "advanced/model"

    def parse_csv(self):
        """Parses the CSV log file into structured data."""
        positive_data = []
        negative_data = []

        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                timestamp, guidance_method, input_guidance, output_guidance, is_negative = row
                entry = {
                    "timestamp": datetime.fromisoformat(timestamp),
                    "guidance_method": guidance_method,
                    "input_guidance": float(input_guidance),
                    "output_guidance": float(output_guidance),
                    "is_negative_conditioning": is_negative.strip().lower() == "true"
                }
                if entry["is_negative_conditioning"]:
                    negative_data.append(entry)
                else:
                    positive_data.append(entry)

        return positive_data, negative_data


    def main(self, anything):
        output_file="output_chart.png"
        positive_data, negative_data = self.parse_csv()

        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Guidance Comparison: Positive vs Negative Conditioning", fontsize=16)

        # Plot Negative Conditioning
        negative_timestamps = [entry["timestamp"] for entry in negative_data]
        negative_input_guidance = [entry["input_guidance"] for entry in negative_data]
        negative_output_guidance = [entry["output_guidance"] for entry in negative_data]
        negative_method = negative_data[0]["guidance_method"] if negative_data else "Unknown"

        ax[0].plot(negative_timestamps, negative_input_guidance, label="Input Guidance", color="blue", linewidth=2)
        ax[0].plot(negative_timestamps, negative_output_guidance, label="Output Guidance", color="green", linewidth=2)
        ax[0].set_title(f"Negative Conditioning ({negative_method})")
        ax[0].set_ylabel("Guidance Value")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        # Plot Positive Conditioning
        positive_timestamps = [entry["timestamp"] for entry in positive_data]
        positive_input_guidance = [entry["input_guidance"] for entry in positive_data]
        positive_output_guidance = [entry["output_guidance"] for entry in positive_data]
        positive_method = positive_data[0]["guidance_method"] if positive_data else "Unknown"

        ax[1].plot(positive_timestamps, positive_input_guidance, label="Input Guidance", color="blue", linewidth=2)
        ax[1].plot(positive_timestamps, positive_output_guidance, label="Output Guidance", color="green", linewidth=2)
        ax[1].set_title(f"Positive Conditioning ({positive_method})")
        ax[1].set_ylabel("Guidance Value")
        ax[1].set_xlabel("Time")
        ax[1].legend(loc="upper right")
        ax[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file)
        plt.close(fig)

        img = Image.open(output_file).convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        output_image = torch.from_numpy(img_array)[None,]
        self.image_cache = output_image

        return (output_image,)
