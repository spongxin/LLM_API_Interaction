from ..clients.base import BaseClient
from ..parsers.pydantic_parser import PydanticOutputParser
from ..prompts.template import PromptTemplate
from pydantic import BaseModel
from typing import Any, List, Dict, Optional
import logging

def run_structured_output_chain(
    client: BaseClient,
    prompt_template: PromptTemplate,
    output_parser: PydanticOutputParser,
    prompt_variables: dict,
    max_retries: int = 3,
    fix_prompt_template: Optional[PromptTemplate] = None,
    **kwargs
) -> tuple[BaseModel, list]:
    """
    Chain for structured output: assemble prompt, call LLM, parse, retry if needed.
    All user prompts (including fix prompts) are constructed via PromptTemplate.
    """
    chat_history: List[Dict[str, str]] = []
    # 1. Assemble prompt with format instructions
    format_instructions = output_parser.get_format_instructions()
    prompt = prompt_template.format(format_instructions=format_instructions, **prompt_variables)
    chat_history.append({"role": "user", "content": prompt})
    # 2. Prepare fix prompt template
    default_fix_tpl = PromptTemplate(
        "Your previous output could not be parsed. Error: $error\n$format_instructions\nPlease correct your output and try again."
    )
    fix_tpl = fix_prompt_template or default_fix_tpl
    # 3. Retry loop
    for _ in range(max_retries):
        response_text = client.chat(messages=chat_history, **kwargs)
        chat_history.append({"role": "assistant", "content": response_text})
        try:
            return output_parser.parse(response_text), chat_history
        except Exception as e:
            fix_prompt = fix_tpl.format(error=str(e), format_instructions=format_instructions)
            chat_history.append({"role": "user", "content": fix_prompt})
            logging.info(f"Parsing error caused by output not following JSON schema.")
    raise RuntimeError(f"Failed to get valid structured output after {max_retries} retries.") 