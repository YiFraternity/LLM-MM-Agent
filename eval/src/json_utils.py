import json
import re
import yaml
from typing import Dict, Any, Optional
from jinja2 import Template, StrictUndefined

def load_yaml(file_path: str) -> Dict:
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def standardize_json(input_text: str, llm_model=None) -> Dict[str, Any]:
    """
    Standardize JSON output using LLM if needed.
    
    Args:
        input_text: Potentially malformed JSON string
        llm_model: Optional LLM model for JSON standardization
        
    Returns:
        Standardized JSON as a dictionary
    """
    if not input_text or not input_text.strip():
        return {}
    
    # First try to parse directly
    try:
        # Clean common JSON issues
        cleaned = re.sub(r',\s*([}\]])', r'\1', input_text)  # Remove trailing commas
        cleaned = re.sub(r'([{\[,])\s*\n\s*([}\],])', r'\1\2', cleaned)  # Remove empty lines in arrays/objects
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if not llm_model:
            return {}
            
        # Load JSON standardization prompt
        prompts = load_yaml('eval/json_standardization.yaml')
        prompt_template = prompts['json_standardization']
        
        # Prepare prompt
        prompt = Template(prompt_template).render(input_text=input_text)
        
        try:
            # Get response from LLM
            response = llm_model.chat(
                [[{"role": "user", "content": prompt}]],
                use_tqdm=False
            )
            
            if response and response[0].outputs:
                standardized_json = response[0].outputs[0].text
                # Extract JSON from markdown code blocks if present
                json_match = re.search(r'```(?:json)?\n([\s\S]*?)\n```', standardized_json)
                if json_match:
                    standardized_json = json_match.group(1)
                
                return json.loads(standardized_json)
        except Exception as e:
            print(f"Error standardizing JSON with LLM: {str(e)}")
            
    return {}

def ensure_json_serializable(data: Any) -> Any:
    """Ensure data is JSON serializable."""
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    elif isinstance(data, dict):
        return {str(k): ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [ensure_json_serializable(item) for item in data]
    else:
        return str(data)
