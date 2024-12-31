# Import the necessary classes
from .translate_guidance_node import (
    TranslateGuidanceSimpleNode,
    TranslateGuidanceStandardNode,
    TranslateGuidanceAdvancedNode,
    TranslateGuidanceChartNode,
)

# Map the node name to the class
NODE_CLASS_MAPPINGS = {
    "TranslateFluxGuidanceSimple": TranslateGuidanceSimpleNode,
    "TranslateFluxGuidanceStandard": TranslateGuidanceStandardNode,  
    "TranslateFluxGuidanceAdvanced": TranslateGuidanceAdvancedNode,  
    "TranslateFluxGuidanceChart": TranslateGuidanceChartNode,
}

# Map a display name to make it look more user-friendly
NODE_DISPLAY_NAME_MAPPINGS = {
    "TranslateFluxGuidanceSimple": "🖼️ Translate Flux Guidance Simple",
    "TranslateFluxGuidanceStandard": "🖼️ Translate Flux Guidance Standard",
    "TranslateFluxGuidanceAdvanced": "🖼️ Translate Flux Guidance Advanced",
    "TranslateFluxGuidanceChart": "📊 Translate Flux Guidance Chart",
}
