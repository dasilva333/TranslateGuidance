from datetime import datetime
from torch import Tensor
import torch
import csv
import os
from .translate_guidance_lib import translate_guidance
from comfy.ldm.flux.layers import (
    timestep_embedding,
)
log_file = os.path.join(os.path.dirname(__file__), "guidance_output.csv")

class TranslateGuidanceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_guidance_method": ([
                    "None",  # Added None/Disabled option
                    "cosine",
                    "inverted_cosine",
                    "sin",
                    "linear_increase",
                    "linear_decrease",
                    "random_noise",
                    "random_gaussian",
                    "random_extreme",
                    "ripsaw",
                    "bubble"
                ],),
                "negative_guidance_method": ([
                    "None",  # Added None/Disabled option
                    "cosine",
                    "inverted_cosine",
                    "sin",
                    "linear_increase",
                    "linear_decrease",
                    "random_noise",
                    "random_gaussian",
                    "random_extreme",
                    "ripsaw",
                    "bubble"
                ],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_transformer_options"
    CATEGORY = "advanced/model"
    
    def initialize_log_file(self):
        """Initialize the CSV log file with headers"""
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Guidance Method', 'Input Guidance', 'Output Guidance', "Negative Conditioning"])

    def log_guidance_values(self, guidance_method, input_guidance, output_guidance, is_negative_conditioning):
        """Append guidance values to the log file"""
        
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

    def apply_transformer_options(self, model, positive_guidance_method, negative_guidance_method):
        # Initialize log file
        self.initialize_log_file()
        
        # Clone the model to avoid modifying the original instance
        m = model.clone()
        is_negative_conditioning = True

        # Replace `forward_orig` with the custom implementation
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
            print("Custom forward_orig has been called")
            
            # Extract patches_replace from transformer_options
            patches_replace = transformer_options.get("patches_replace", {})
            nonlocal is_negative_conditioning  # Ensure access to the flag from the outer scope

            # Validate input dimensions
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            # Running on sequences img
            img = m.model.diffusion_model.img_in(img)
            vec = m.model.diffusion_model.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

            # Handle guidance embedding
            if m.model.diffusion_model.params.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                
                # Check if guidance_method is set in transformer_options
                if (is_negative_conditioning):
                    current_guidance_method = transformer_options.get("negative_guidance_method", None)
                else:
                    current_guidance_method = transformer_options.get("positive_guidance_method", None)
                
                # Process and log guidance values
                if current_guidance_method and current_guidance_method != "None":
                    print(f"Using guidance method: {current_guidance_method} for guidance: {guidance}")
                    translated_guidance = translate_guidance(timesteps, guidance, img.device, current_guidance_method)
                    output_guidance = translated_guidance
                    vec = vec + m.model.diffusion_model.guidance_in(timestep_embedding(translated_guidance, 256).to(img.dtype))
                else:
                    output_guidance = timestep_embedding(guidance, 256)
                    vec = vec + m.model.diffusion_model.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
                
                # Log the guidance values
                self.log_guidance_values(
                    current_guidance_method,
                    guidance,
                    output_guidance.mean().item(),
                    is_negative_conditioning
                )

            vec = vec + m.model.diffusion_model.vector_in(y[:, :m.model.diffusion_model.params.vec_in_dim])
            txt = m.model.diffusion_model.txt_in(txt)

            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = m.model.diffusion_model.pe_embedder(ids)

            # Process double stream blocks
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
                
                if control is not None:  # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((txt, img), 1)

            # Process single stream blocks
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
                
                if control is not None:  # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1]:, ...] += add

            img = img[:, txt.shape[1]:, ...]

            # Final layer processing
            img = m.model.diffusion_model.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

            # Toggle the flag at each call
            is_negative_conditioning = not is_negative_conditioning

            # Optionally log the current state
            print(f"Current conditioning: {'Negative' if is_negative_conditioning else 'Positive'}")

            return img

        # Override the `forward_orig` method
        m.model.diffusion_model.forward_orig = custom_forward_orig
        print("Custom forward_orig applied to diffusion_model.")

        # Ensure model_options exists
        if not hasattr(m, "model_options") or not isinstance(m.model_options, dict):
            m.model_options = {}

        # Check or initialize transformer_options
        if "transformer_options" not in m.model_options:
            m.model_options["transformer_options"] = {}

        # Update transformer_options with the selected guidance method only if it's not None
        if positive_guidance_method != "None":
            m.model_options["transformer_options"].update({"positive_guidance_method": positive_guidance_method})
        else:
            # Remove the guidance_method key if it exists when None is selected
            m.model_options["transformer_options"].pop("positive_guidance_method", None)

        if negative_guidance_method != "None":
            m.model_options["transformer_options"].update({"negative_guidance_method": negative_guidance_method})
        else:
            # Remove the guidance_method key if it exists when None is selected
            m.model_options["transformer_options"].pop("negative_guidance_method", None)    

        print("Updated transformer_options:", m.model_options["transformer_options"])
        return (m,)