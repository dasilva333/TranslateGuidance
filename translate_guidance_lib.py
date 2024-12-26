import torch
import math

def inverted_cosine_bubble_ripsaw(timestep, guidance, device, adjust_normal_cfg=True):
    # Ensure timestep is [0,1].
    t = timestep.to(device, dtype=torch.float32).clamp(0.0, 1.0)

    # Original guidance as float on the correct device
    g = guidance.to(device, dtype=torch.float32)
    g2 = None
    transformed2 = None

    # If we have more than one channel and want to adjust normal cfg
    if adjust_normal_cfg and len(g.shape) > 1:
        g = g[:, 0]
        g2 = g[:, 1]

    # Generate random shifts (0 to 8 inclusive)
    # Cast guidance to int and shift, then cast back to float
    g_int = g.to(torch.int64)  # safer for bit-shift
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

    # Debug prints
    # print("Random bit-shift bubble ripsaw transformation:")
    # print("Original guidance:", g)
    # print("Bit shifts used:", shift if g2 is None else (shift, shift2))
    # print("Guidance after shifting:", g_shifted if g2 is None else (g_shifted, g2_shifted))
    # print("Transformed guidance:", transformed)
    # print("Linear progress (t):", t)
    # print("Resulting guidance:", out)

    return out

def inverted_cosine_ripsaw(timestep, guidance, device, adjust_normal_cfg=True):
    """
    Similar to 'inverted_cosine' but with a random bit-shift ripsaw approach.

    We'll randomly choose an integer shift ∈ [0..8], cast guidance to int, left shift it,
    cast back to float, and use that in our inverted_cosine formula. The result then
    gets linearly blended with the original guidance based on the timestep 't'.

    Args:
      timestep : scalar or 1D tensor in [0,1], representing the normalized progress.
      guidance : scalar or 1D tensor containing the original guidance value(s).
      device   : torch device on which computations will run.
      adjust_normalcfg (bool) : if True and guidance is multi-dimensional, we
                                 adjust the first dimension(s) for dual-step transforms.

    Returns:
      out : The transformed guidance after random bit-shift peaks, mapped via
            an inverted cosine, and linearly interpolated against the original guidance.
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
        g2 = g[:, 1]

    # Generate random shifts (0 to 8 inclusive)
    # Cast guidance to int and shift, then cast back to float
    g_int = g.to(torch.int64)  # safer for bit-shift
    shift = torch.randint(low=0, high=255, size=g_int.shape, device=device, dtype=torch.int64)
    g_shifted = (g_int << shift).to(torch.float32)

    # Inverted cosine with the shifted guidance
    transformed = 4.5 - torch.cos(math.pi * (g_shifted + t))

    if g2 is not None:
        g2_int = g2.to(torch.int64)
        shift2 = torch.randint(low=0, high=9, size=g2_int.shape, device=device, dtype=torch.int64)
        g2_shifted = (g2_int << shift2).to(torch.float32)
        transformed2 = 4.5 - torch.cos(math.pi * (g2_shifted + t) * 0.95)

    if g2 is not None and transformed2 is not None:
        transformed = torch.stack([transformed, transformed2], dim=-1)

    # Interpolate between original guidance (g) and the new transformed value
    out = (1.0 - t) * g + (t * transformed)

    # Debug prints
    # print("Random bit-shift ripsaw transformation:")
    # print("Original guidance:", g)
    # print("Bit shifts used:", shift if g2 is None else (shift, shift2))
    # print("Guidance after shifting:", g_shifted if g2 is None else (g_shifted, g2_shifted))
    # print("Transformed guidance:", transformed)
    # print("Linear progress (t):", t)
    # print("Resulting guidance:", out)

    return out

def translate_guidance(timestep, guidance, device, method="inverted_cosine"):
    """
    Transform guidance strength based on a normalized 'timestep' in [0,1].
    """
    t = timestep.to(device, dtype=torch.float32).clamp_(0.0, 1.0)
    g = guidance.to(device, dtype=torch.float32)

    if method == "cosine":
        transformed = torch.cos(math.pi * (g + t))
    elif method == "inverted_cosine":
        transformed = 5.0 - torch.cos(math.pi * (g + t))
    elif method == "sin":
        transformed = torch.sin(math.pi * (g + t))
    elif method == "linear_increase":
        transformed = g + (1.5 * t)
    elif method == "linear_decrease":
        transformed = g - (1.5 * t)
    elif method == "random_noise":
        noise = torch.empty_like(g).uniform_(-0.75, 0.75)
        transformed = g + (noise * t)
    elif method == "random_gaussian":
        noise = torch.randn_like(g) * 0.3
        transformed = g + (noise * t)
    elif method == "random_extreme":
        noise = torch.empty_like(g).uniform_(-2.0, 2.0)
        transformed = g + (noise * t)
    elif method == "ripsaw":
        transformed = inverted_cosine_ripsaw(timestep, guidance, device)
    elif method == "bubble":
        transformed = inverted_cosine_bubble_ripsaw(timestep, guidance, device)
    else:
        transformed = g

    out = (1.0 - t) * g + (t * transformed)
    return out
