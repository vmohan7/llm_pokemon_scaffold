import copy

from typing import Any

from google.genai import types

def convert_to_deepseek_format(message_history):
    """
    Converts message history to DeepSeek format where content is a string, handling content as either a string
    or an array of objects, extracting text from objects with type 'text', preserving the original order.
    
    Args:
        message_history (list): List of dictionaries with 'role' and 'content' where content is either
                               a string or an array of {'type': string, 'text': string} objects
    
    Returns:
        list: New list with 'role' and 'content' as a string, only for messages with valid text content,
              in original order
    """
    deepseek_history = []
    
    for msg in message_history:
        if not isinstance(msg.get("content"), (str, list)):
            continue
            
        if isinstance(msg["content"], str):
            deepseek_history.append({"role": msg["role"], "content": msg["content"]})
        else:  # content is a list
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    deepseek_history.append({"role": msg["role"], "content": item["text"]})
                    break  # Take only the first valid text item
                
    return deepseek_history

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
        x["function"]= {"name": x["name"], "description": x["description"], "parameters": x["input_schema"]}
        del x["name"]
        del x["description"]
        del x["input_schema"]
        x["function"]["parameters"]["properties"]["explanation_of_action"] = {
                        "type": "string",
                        "description": "MANDATORY: A detailed explanation of why you called this tool"
                    }
        x["function"]["parameters"]["required"].append("explanation_of_action")
        x["type"] = "function"
        OPENAI_TOOLS.append(x)
    return OPENAI_TOOLS