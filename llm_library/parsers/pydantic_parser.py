from pydantic import BaseModel, ValidationError
from typing import Type, Optional
import json
import json_repair
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
        json_object = json_repair.loads(text)
        try:
            return self.model.model_validate(json_object)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse or validate JSON: {e}") from e 