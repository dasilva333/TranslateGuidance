# Flux Enhanced Guidance Methods

This repository enhances the Flux model with additional guidance transformation methods for improved control and creativity during image generation.

## Features

- **Extended Guidance Methods**: 10 methods including cosine, inverted cosine, sin, linear increase, random noise, ripsaw, and bubble.
- **Enhanced Creativity**: Provides a greater range of CFG values for Flux-guided samplers.
- **Integration**: Works seamlessly with FluxGuidance for the best results.
- **Dynamic Control**: Offers a variety of transformations, from smooth (cosine) to sharp and dynamic (ripsaw, bubble).

## Installation

1. **Copy Files Manually** to this directory `custom_nodes\TranslateGuidance`
2. In the ComfyUI do `Manager` > `Install Git Url` > `Copy the link for this project` no requirements.txt needed

## Notes

- This only affects images generated with the Flux model.
- Enables greater range and control of CFG values on the sampler.
- Works best when paired with `FluxGuidance`.
- Recommended methods:
  - **Inverted Cosine**: Smooth and consistent transformations.
  - **Ripsaw**: Sharp transitions and dramatic peaks.
  - **Bubble**: High-contrast and dynamic outputs.
- **Random Extreme**: Ideal for pseudo-SDE-like generations.

## Available Guidance Methods

| Method             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **cosine**         | Smooth wave-like transitions using cosine function.                        |
| **inverted_cosine**| Reverse cosine for inverted, smooth transitions.                           |
| **sin**            | Periodic sine wave adjustments.                                            |
| **linear_increase**| Gradual linear increase in guidance strength.                              |
| **linear_decrease**| Gradual linear decrease in guidance strength.                              |
| **random_noise**   | Adds uniform random noise for slight variation.                            |
| **random_gaussian**| Adds Gaussian noise for natural randomness.                                |
| **random_extreme** | Adds extreme random noise for dramatic variations.                         |
| **ripsaw**         | Random bit-shift followed by inverted cosine for sharp peaks.              |
| **bubble**         | Dual-channel sharp peaks with alternating shifts for dynamic outputs.      |

## Usage

1. Select your preferred **guidance_method** when configuring Flux:
   - Available methods include cosine, inverted cosine, ripsaw, bubble, and more.
2. Pair with **FluxGuidance** for optimal results.
3. Experiment with different methods to achieve desired effects.

Happy creating!
