import re
from string import Template

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
        # 兼容旧的{var}风格，自动转换为$var
        self.input_variables = re.findall(r"\{(\w+)\}", template)
        # 将{var}替换为$var，避免与json大括号冲突
        self._compiled_template = Template(re.sub(r"\{(\w+)\}", r"$\1", template))

    def format(self, **kwargs) -> str:
        return self._compiled_template.substitute(**kwargs) 