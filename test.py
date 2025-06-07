import unittest
from llm_library.prompts.template import PromptTemplate
from llm_library.parsers.pydantic_parser import PydanticOutputParser
from llm_library.chains.structured_output import run_structured_output_chain
from llm_library.clients.providers import register_client, get_client
from pydantic import BaseModel, Field
from typing import List

# Pydantic model for testing
class TestModel(BaseModel):
    foo: str
    bar: int

# DummyClient for chain tests
@register_client("dummy")
class DummyClient:
    def __init__(self, endpoint=None, api_key=None, model_name=None, **kwargs):
        self.responses = kwargs.get('responses', [])
        self.call_count = 0
    def chat(self, messages, **kwargs):
        resp = self.responses[min(self.call_count, len(self.responses)-1)]
        self.call_count += 1
        return resp

class TestPromptTemplate(unittest.TestCase):
    def test_format(self):
        tpl = PromptTemplate("Hello {name}, your score is {score}.")
        result = tpl.format(name="Alice", score=95)
        self.assertEqual(result, "Hello Alice, your score is 95.")

class TestPydanticOutputParser(unittest.TestCase):
    def test_parse_valid_json(self):
        parser = PydanticOutputParser(TestModel)
        text = '{"foo": "abc", "bar": 123}'
        obj = parser.parse(text)
        self.assertEqual(obj.foo, "abc")
        self.assertEqual(obj.bar, 123)
    def test_parse_json_in_markdown(self):
        parser = PydanticOutputParser(TestModel)
        text = '```json\n{"foo": "xyz", "bar": 456}\n```'
        obj = parser.parse(text)
        self.assertEqual(obj.foo, "xyz")
        self.assertEqual(obj.bar, 456)
    def test_parse_invalid_json(self):
        parser = PydanticOutputParser(TestModel)
        with self.assertRaises(ValueError):
            parser.parse('not a json')
    def test_parse_schema_validation_error(self):
        parser = PydanticOutputParser(TestModel)
        # bar should be int, not str
        with self.assertRaises(ValueError):
            parser.parse('{"foo": "abc", "bar": "not_an_int"}')

class TestStructuredOutputChain(unittest.TestCase):
    def test_chain_with_retry(self):
        prompt = PromptTemplate("Give me a JSON. {format_instructions}")
        parser = PydanticOutputParser(TestModel)
        # First response is invalid, second is valid
        client = DummyClient(responses=["not a json", '{"foo": "ok", "bar": 2}'])
        result, history = run_structured_output_chain(
            client=client,
            prompt_template=prompt,
            output_parser=parser,
            prompt_variables={},
            max_retries=2
        )
        self.assertEqual(result.foo, "ok")
        self.assertEqual(result.bar, 2)
        self.assertEqual(len(history), 4)
    def test_chain_fail(self):
        prompt = PromptTemplate("Give me a JSON. {format_instructions}")
        parser = PydanticOutputParser(TestModel)
        client = DummyClient(responses=["bad", "still bad"])
        with self.assertRaises(RuntimeError):
            run_structured_output_chain(
                client=client,
                prompt_template=prompt,
                output_parser=parser,
                prompt_variables={},
                max_retries=2
            )

if __name__ == "__main__":
    unittest.main() 