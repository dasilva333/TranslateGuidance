import torch
import math


# -----------------------------------
# FORMULAS + PATTERNS
# -----------------------------------

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


# -------------------------------------------------------------------
# Example 1: Two specialized transformations used by pattern formulas
# -------------------------------------------------------------------
def inverted_cosine_bubble_ripsaw(timestep, guidance, device, adjust_normal_cfg=True):
    """
    An example method that modifies guidance by applying
    an inverted cosine with random bit-shifts, producing
    a 'bubble' or 'ripsaw' style effect.
    """
    # Ensure timestep is [0,1].
    t = timestep.to(device, dtype=torch.float32).clamp(0.0, 1.0)

    # Original guidance as float on the correct device
    g = guidance.to(device, dtype=torch.float32)
    g2 = None
    transformed2 = None

    # If we have more than one channel and want to adjust normal cfg
    if adjust_normal_cfg and len(g.shape) > 1:
        g = g[:, 0]
        g2 = guidance[:, 1]

    # Generate random shifts (0 to 8 inclusive)
    g_int = g.to(torch.int64)
    shift = torch.randint(low=1, high=8, size=g_int.shape, device=device, dtype=torch.int64)
    g_shifted = (g_int << shift).to(torch.float32)

    # Inverted cosine with the shifted guidance
    transformed = 4.5 - torch.cos(math.pi * (g_shifted + t) * 0.9995)

    if g2 is not None:
        g2_int = g2.to(torch.int64)
        shift2 = torch.randint(low=-8, high=-1, size=g2_int.shape, device=device, dtype=torch.int64)
        g2_shifted = (g2_int << shift2).to(torch.float32)
        transformed2 = 4.5 - torch.cos(math.pi * (g2_shifted + t) * 0.9995)

    if g2 is not None and transformed2 is not None:
        transformed = torch.stack([transformed, transformed2], dim=-1)

    # Interpolate between original guidance (g) and the new transformed value
    out = (1.0 - t) * g + (t * transformed)
    return out


def inverted_cosine_ripsaw(timestep, guidance, device, adjust_normal_cfg=True):
    """
    Similar approach to 'inverted_cosine_bubble_ripsaw', but with different
    shift ranges to produce a more chaotic 'ripsaw' effect.
    """
    # Ensure timestep is [0,1].
    t = timestep.to(device, dtype=torch.float32).clamp(0.0, 1.0)

    # Original guidance as float on the correct device
    g = guidance.to(device, dtype=torch.float32)
    g2 = None
    transformed2 = None

    # If we have more than one channel and want to adjust normal cfg
    if adjust_normal_cfg and len(g.shape) > 1:
        g = g[:, 0]
        g2 = guidance[:, 1]

    g_int = g.to(torch.int64)
    shift = torch.randint(low=0, high=31, size=g_int.shape, device=device, dtype=torch.int64)
    g_shifted = (g_int << shift).to(torch.float32)

    transformed = 4.5 - torch.cos(math.pi * (g_shifted + t))

    if g2 is not None:
        g2_int = g2.to(torch.int64)
        shift2 = torch.randint(low=0, high=31, size=g2_int.shape, device=device, dtype=torch.int64)
        g2_shifted = (g2_int << shift2).to(torch.float32)
        transformed2 = 4.5 - torch.cos(math.pi * (g2_shifted + t) * 0.95)

    if g2 is not None and transformed2 is not None:
        transformed = torch.stack([transformed, transformed2], dim=-1)

    # Interpolate between original guidance (g) and the new transformed value
    out = (1.0 - t) * g + (t * transformed)
    return out


# -------------------------------------------------------------
# 1) Primary Formula Translator
# -------------------------------------------------------------
def translate_formula(timestep: torch.Tensor,
                      guidance: torch.Tensor,
                      device: torch.device,
                      formula_name: str = "None",
                      base_value: float = 4.5) -> torch.Tensor:
    """
    Applies a 'formula' transformation to guidance, e.g. cosine, sin, random, etc.

    Args:
        timestep    : scalar or 1D tensor in [0,1], representing the normalized progress.
        guidance    : scalar or 1D tensor containing the original guidance value(s).
        device      : torch device on which computations will run.
        formula_name: one of the keys from FORMULAS list, e.g. 'cosine', 'inverted_cosine', etc.
        base_value  : a baseline offset for inverted trig transforms.

    Returns:
        A transformed guidance tensor, same shape as input `guidance`.
    """

    # Convert to float on the correct device
    t = timestep.to(device, dtype=torch.float32).clamp_(0.0, 1.0)
    g = guidance.to(device, dtype=torch.float32)
    g2 = None
    if g.dim() > 1:
        # If we have multiple guidance channels (e.g. for normal CFG + something else),
        # you can split them here or keep them stacked.
        g = g[:, 0]
        g2 = guidance[:, 1]

    if formula_name == "None":
        # No transformation
        transformed = g

    elif formula_name == "cosine":
        transformed = torch.cos(math.pi * (g + t))

    elif formula_name == "inverted_cosine":
        transformed = base_value - torch.cos(math.pi * (g + t))

    elif formula_name == "sin":
        transformed = torch.sin(math.pi * (g + t))

    elif formula_name == "inverted_sin":
        transformed = base_value - torch.sin(math.pi * (g + t))

    elif formula_name == "linear_increase":
        # Increases linearly with t
        slope = 1.5
        transformed = g + (slope * t)

    elif formula_name == "linear_decrease":
        # Decreases linearly with t
        slope = 1.5
        transformed = g - (slope * t)

    elif formula_name == "random_noise":
        # Uniform random noise in [-0.75, 0.75]
        noise = torch.empty_like(g).uniform_(-0.75, 0.75)
        transformed = g + (noise * t)

    elif formula_name == "random_gaussian":
        # Gaussian random noise with std=0.3
        noise = torch.randn_like(g) * 0.3
        transformed = g + (noise * t)

    elif formula_name == "random_extreme":
        # Larger uniform random range
        noise = torch.empty_like(g).uniform_(-2.0, 2.0)
        transformed = g + (noise * t)

    # For more advanced formula logic, add them here:
    # - "intelligent_up"
    # - "intelligent_down"
    # - "intelligent_oscillation"
    # - "intelligent_random"
    #
    # ... Placeholder:
    elif formula_name.startswith("intelligent_"):
        # E.g. you might have a more sophisticated function
        # that depends on the current iteration (like a PID).
        # For now, let’s just pass the guidance as-is.
        transformed = g

    else:
        # Default if no match
        transformed = g

    # If there's a second channel, transform it the same or differently
    if g2 is not None:
        # For simplicity, we re-use the same transform on g2
        # If you want a separate approach, you can replicate the above code for g2
        # or do something special here.
        if formula_name == "None":
            transformed2 = g2
        elif formula_name == "cosine":
            transformed2 = torch.cos(math.pi * (g2 + t))
        elif formula_name == "inverted_cosine":
            transformed2 = base_value - torch.cos(math.pi * (g2 + t))
        # More cases for g2 if needed...
        else:
            # Fall back for unimplemented formula on second channel
            transformed2 = g2

        # Re-stack (N, 2) if you had 2 channels initially
        transformed = torch.stack([transformed, transformed2], dim=-1)

    return transformed


# -------------------------------------------------------------
# 2) Primary Pattern Translator
# -------------------------------------------------------------
def translate_pattern(timestep: torch.Tensor,
                      guidance: torch.Tensor,
                      device: torch.device,
                      pattern_name: str = "None") -> torch.Tensor:
    """
    Applies a 'pattern' transformation on top of the guidance.
    Example patterns: 'ripsaw', 'bubble', 'ripple', etc.

    Args:
        timestep    : scalar or 1D tensor in [0,1].
        guidance    : scalar or 1D tensor of original guidance.
        device      : torch device on which computations will run.
        pattern_name: one of the keys from PATTERN, e.g. 'chainsaw', 'ripsaw', 'bubble', etc.

    Returns:
        A further transformed guidance tensor.
    """

    if pattern_name == "None":
        return guidance  # no change

    if pattern_name == "ripsaw":
        return inverted_cosine_ripsaw(timestep, guidance, device)

    elif pattern_name == "bubble":
        return inverted_cosine_bubble_ripsaw(timestep, guidance, device)

    # Additional placeholders for other pattern logic
    # 'chainsaw', 'ripple', 'shockwave', 'pinetree', 'cascade'
    #
    elif pattern_name == "chainsaw":
        # You could implement a sawtooth wave or any other approach
        # For demonstration, do a very simple transformation:
        return guidance * (1.0 + 0.5 * torch.sin(2.0 * math.pi * timestep))

    elif pattern_name == "ripple":
        # Another placeholder
        ripple_factor = torch.sin(4.0 * math.pi * timestep) * 0.25
        return guidance + ripple_factor

    else:
        # If not implemented, just return guidance
        return guidance


# --------------------------------------------------------------------
# MAIN: translate_guidance (combining formula + pattern translators)
# --------------------------------------------------------------------
def translate_guidance(timestep: torch.Tensor,
                       guidance: torch.Tensor,
                       device: torch.device,
                       config: dict = None) -> torch.Tensor:
    """
    The main method that uses both the formula translator and pattern translator.
    It first applies the chosen formula, then the chosen pattern, returning the result.

    Args:
        timestep: scalar or 1D tensor in [0,1].
        guidance: scalar or 1D tensor for the original guidance.
        device  : torch device on which computations will run.
        config  : dict containing keys like:
                    {
                      'positive_guidance_formula': 'cosine',  # or 'None', 'inverted_cosine', etc.
                      'positive_pattern': 'ripsaw',           # or 'None', 'bubble', etc.
                      'positive_lowest_guidance': ...,
                      'positive_median_guidance': ...,
                      'positive_highest_guidance': ...,
                      'positive_guidance_scale': ...,
                      ...
                    }

    Returns:
        The transformed guidance after applying formula + pattern.
    """

    if config is None:
        config = {}

    # Grab relevant formula/pattern names from the config
    formula_name = config.get('positive_guidance_formula', 'None')
    pattern_name = config.get('positive_pattern', 'None')

    # Step 1: Apply formula
    formula_transformed = translate_formula(
        timestep,
        guidance,
        device,
        formula_name=formula_name,
        base_value=4.5  # or whatever default base offset you want
    )

    # Step 2: Apply pattern
    patterned_transformed = translate_pattern(
        timestep,
        formula_transformed,
        device,
        pattern_name=pattern_name
    )

    # If you want to do something with positive_lowest_guidance, etc., you can incorporate it here:
    # Example: clamp the final result between certain min/max
    # min_val = config.get('positive_lowest_guidance', 1.0)
    # max_val = config.get('positive_highest_guidance', 6.0)
    # patterned_transformed = patterned_transformed.clamp_(min_val, max_val)

    # If you want an overall scale:
    scale_val = config.get('positive_guidance_scale', 1.0)
    final = patterned_transformed * scale_val

    return final
