from typing import Literal

# Original configuration for the application. Unfortunately I am not very disciplined so only a small amount of settings are here.
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"
SAMBANOVA_VISION_MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct"
SAMBANOVA_TOOL_MODEL_ID = "DeepSeek-V3-0324"
SAMBANOVA_STRATEGIST_MODEL_ID = "DeepSeek-R1"

TEMPERATURE = 1.0
MAX_TOKENS = 10000

# bypass using Claude for "navigate_to_offscreen_coordinate" because it's really token-expensive and also we know it can reliably do it if we give a HUGE amount of tokens,
# so it doesn't prove much. (plus I'd have to do streaming which is just annoying)
# This basically saves money at the cost of being a bit unsatisfying. This is a lot faster though.
DIRECT_NAVIGATION = True