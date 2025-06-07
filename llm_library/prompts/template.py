import re

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
        self.input_variables = re.findall(r"\{(\w+)\}", template)

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs) 