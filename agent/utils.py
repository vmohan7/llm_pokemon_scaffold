import copy

from typing import Any

from google.genai import types


def convert_tool_defs_to_google_format(tool_defs: list[dict[str, Any]]) -> list[types.Tool]:
    GOOGLE_TOOLS: list[types.Tool] = []
    for tool_desc in tool_defs:
        x = copy.copy(tool_desc)  # Shallow copy should be fine here
        x["parameters"] = x["input_schema"]
        del x["input_schema"]
        GOOGLE_TOOLS.append(types.Tool(function_declarations=[x]))
    return GOOGLE_TOOLS

def convert_tool_defs_to_openai_format(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    OPENAI_TOOLS = []
    for tool_desc in tool_defs:
        x = copy.deepcopy(tool_desc)
        x["parameters"] = x["input_schema"]
        del x["input_schema"]
        x["parameters"]["properties"]["explanation_of_action"] = {
                        "type": "string",
                        "description": "MANDATORY: A detailed explanation of why you called this tool"
                    }
        x["parameters"]["required"].append("explanation_of_action")
        x["type"] = "function"
        OPENAI_TOOLS.append(x)
    return OPENAI_TOOLS