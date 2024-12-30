from datetime import datetime
from torch import Tensor
import torch
import csv
import os

# Import your custom translator; ensure that it can accept
# formula_name, pattern_name, min_guidance, mid_guidance, max_guidance,
# formula_scale, pattern_scale, is_negative, etc.
from .translate_guidance_lib import translate_guidance

from comfy.ldm.flux.layers import (
    timestep_embedding,
)

# CSV log file path
log_file = os.path.join(os.path.dirname(__file__), "guidance_output.csv")

# Guidance formula + pattern lists
FORMULAS = [
    "None",  # Added None/Disabled option
    "cosine",
    "inverted_cosine",
    "sin",
    "inverted_sin",
    "linear_increase",
    "linear_decrease",
    "random_noise",
    "random_gaussian",
    "random_extreme",
    "intelligent_up",
    "intelligent_down",
    "intelligent_oscillation",
    "intelligent_random",
]

PATTERN = [
    "None",  # Added None/Disabled option
    "chainsaw",
    "ripsaw",
    "bubble",
    "ripple",
    "shockwave",
    "pinetree",
    "cascade",
]


class AdvancedTranslateGuidanceNode:
    def __init__(self):
        pass


class TranslateGuidanceNode:
    @classmethod
    def INPUT_TYPES(cls):
        """
        Describes the input types for the node.
        """
        return {
            "required": {
                "model": ("MODEL",),

                "positive_guidance_formula": (FORMULAS, {"default": "None"}),
                "positive_pattern": (PATTERN, {"default": "None"}),
                "positive_lowest_guidance": ("FLOAT", {"default": 0.5, "min": 0.0}),
                "positive_median_guidance": ("FLOAT", {"default": 3.5, "min": 0.0}),
                "positive_highest_guidance": ("FLOAT", {"default": 6.0, "min": 0.0}),
                "positive_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "positive_pattern_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),

                "negative_guidance_formula": (FORMULAS, {"default": "None"}),
                "negative_pattern": (PATTERN, {"default": "None"}),
                "negative_lowest_guidance": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "negative_median_guidance": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "negative_highest_guidance": ("FLOAT", {"default": 2.0, "min": 0.0}),
                "negative_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "negative_pattern_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_transformer_options"
    CATEGORY = "advanced/model"

    def initialize_log_file(self):
        """
        Initialize the CSV log file with headers
        """
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Guidance Method',
                'Input Guidance',
                'Output Guidance',
                'Negative Conditioning'
            ])

    def log_guidance_values(self, guidance_method, input_guidance, output_guidance, is_negative_conditioning):
        """
        Append guidance values to the log file
        """
        # Convert tensor values to float if necessary
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

    def apply_transformer_options(self,
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
                                  negative_pattern_scale):
        """
        Main entry point for customizing the model with a custom forward method
        that modifies guidance based on formulas/patterns.
        """
        # 1) Initialize log file
        self.initialize_log_file()

        # 2) Clone the model to avoid modifying the original instance
        m = model.clone()

        # 3) This flag toggles each call (negative vs. positive)
        #    We'll start in negative mode so that the first pass will get negative formula,
        #    then it flips to positive formula, etc.
        is_negative_conditioning = True

        # --------------------------
        #   Custom forward method
        # --------------------------
        def custom_forward_orig(
                img: Tensor,
                img_ids: Tensor,
                txt: Tensor,
                txt_ids: Tensor,
                timesteps: Tensor,
                y: Tensor,
                guidance: Tensor = None,
                control=None,
                transformer_options={},
                attn_mask: Tensor = None,
        ) -> Tensor:
            nonlocal is_negative_conditioning

            # Extract patches_replace from transformer_options (for advanced usage)
            patches_replace = transformer_options.get("patches_replace", {})

            # Validate input dimensions
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            # 1) Input transformations
            img = m.model.diffusion_model.img_in(img)
            vec = m.model.diffusion_model.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

            # 2) Guidance embedding
            if m.model.diffusion_model.params.guidance_embed:
                if guidance is None:
                    raise ValueError("No guidance strength found for this guidance-distilled model.")

                # Decide whether to use positive or negative formula/pattern this pass
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

                # If "None", skip custom transform entirely; otherwise, call translate_guidance
                if current_formula and current_formula != "None":
                    # Prepare config for translate_guidance
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
                    output_guidance = translated_guidance

                    # Add guidance embedding
                    vec = vec + m.model.diffusion_model.guidance_in(
                        timestep_embedding(translated_guidance, 256).to(img.dtype)
                    )
                else:
                    # Default behavior if no transformation
                    output_guidance = timestep_embedding(guidance, 256)
                    vec = vec + m.model.diffusion_model.guidance_in(output_guidance.to(img.dtype))

                # Log the guidance values (using mean for output tensor)
                self.log_guidance_values(
                    guidance_method=current_formula,
                    input_guidance=guidance,
                    output_guidance=output_guidance.mean().item(),
                    is_negative_conditioning=is_negative_conditioning
                )

            # 3) Combine other embeddings
            vec = vec + m.model.diffusion_model.vector_in(y[:, :m.model.diffusion_model.params.vec_in_dim])
            txt = m.model.diffusion_model.txt_in(txt)

            # 4) Prepare positional embeddings
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = m.model.diffusion_model.pe_embedder(ids)

            # 5) Double-stream blocks
            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                if ("double_block", i) in patches_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(
                            img=args["img"],
                            txt=args["txt"],
                            vec=args["vec"],
                            pe=args["pe"],
                            attn_mask=args.get("attn_mask")
                        )
                        return out

                    out = patches_replace[("double_block", i)](
                        {"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask},
                        {"original_block": block_wrap}
                    )
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(
                        img=img,
                        txt=txt,
                        vec=vec,
                        pe=pe,
                        attn_mask=attn_mask
                    )

                # If we have control inputs, apply them
                if control is not None:
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            # Merge text + image streams
            img = torch.cat((txt, img), 1)

            # 6) Single-stream blocks
            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                if ("single_block", i) in patches_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(
                            args["img"],
                            vec=args["vec"],
                            pe=args["pe"],
                            attn_mask=args.get("attn_mask")
                        )
                        return out

                    out = patches_replace[("single_block", i)](
                        {"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask},
                        {"original_block": block_wrap}
                    )
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

                # ControlNet outputs
                if control is not None:
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            # add only to the latter portion (image tokens)
                            img[:, txt.shape[1]:, ...] += add

            # Extract out the image portion
            img = img[:, txt.shape[1]:, ...]

            # 7) Final layer
            img = m.model.diffusion_model.final_layer(img, vec)

            # Toggle negative <-> positive
            is_negative_conditioning = not is_negative_conditioning

            return img

        # -----------------------------------------------------------------
        # END custom_forward_orig
        # -----------------------------------------------------------------

        # Override original forward method
        m.model.diffusion_model.forward_orig = custom_forward_orig
        print("Custom forward_orig applied to diffusion_model.")

        # Make sure model_options exists
        if not hasattr(m, "model_options") or not isinstance(m.model_options, dict):
            m.model_options = {}

        # Check or initialize 'transformer_options'
        if "transformer_options" not in m.model_options:
            m.model_options["transformer_options"] = {}

        # -------------------------------------------------
        # Update the transformer's guidance config (positive)
        # -------------------------------------------------
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
            # Remove if it's set to None
            m.model_options["transformer_options"].pop("positive_guidance_formula", None)
            m.model_options["transformer_options"].pop("positive_pattern", None)
            m.model_options["transformer_options"].pop("positive_lowest_guidance", None)
            m.model_options["transformer_options"].pop("positive_median_guidance", None)
            m.model_options["transformer_options"].pop("positive_highest_guidance", None)
            m.model_options["transformer_options"].pop("positive_guidance_scale", None)
            m.model_options["transformer_options"].pop("positive_pattern_scale", None)

        # -------------------------------------------------
        # Update the transformer's guidance config (negative)
        # -------------------------------------------------
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

        # Print final options for debugging
        print("Updated transformer_options:", m.model_options["transformer_options"])

        return (m,)
