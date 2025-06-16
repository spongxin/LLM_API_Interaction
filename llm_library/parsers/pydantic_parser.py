from pydantic import BaseModel, ValidationError
from typing import Type, Optional
import json
import re
from llm_library.prompts.template import PromptTemplate

class PydanticOutputParser:
    DEFAULT_FORMAT_TEMPLATE = PromptTemplate(
        "Your output must be a JSON object that matches the following Pydantic schema:\n```json\n$schema\n```"
    )

    def __init__(self, pydantic_model: Type[BaseModel], format_template: Optional[PromptTemplate] = None):
        self.model = pydantic_model
        self.format_template = format_template or self.DEFAULT_FORMAT_TEMPLATE

    def get_format_instructions(self) -> str:
        """Generate JSON schema format instructions using PromptTemplate."""
        schema = json.dumps(self.model.model_json_schema(), indent=2)
        return self.format_template.format(schema=schema)

    def parse(self, text: str) -> BaseModel:
        """Extract, validate, and parse JSON from LLM output."""
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        match = re.search(r"```json\n({.*?})\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start_pos = text.find('{')
            if start_pos == -1:
                raise ValueError("No JSON object found in the output.")
            # 从第一个{开始，找到匹配的最后一个}
            stack = []
            for i in range(start_pos, len(text)):
                if text[i] == '{':
                    stack.append(i)
                elif text[i] == '}':
                    if stack:
                        stack.pop()
                        if not stack:  # 找到匹配的最后一个}
                            end_pos = i
                            break
            else:
                raise ValueError("No valid JSON object found in the output.")
            json_str = text[start_pos:end_pos+1]
        try:
            json_object = json.loads(json_str)
            return self.model.model_validate(json_object)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse or validate JSON: {e}") from e 