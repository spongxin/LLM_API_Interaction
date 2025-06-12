from llm_library import get_client, PydanticOutputParser, PromptTemplate, run_structured_output_chain
from pydantic import BaseModel, Field
from typing import List
import logging

logging.basicConfig(level=logging.INFO)

# 1. Define output structure
class FinalDecision(BaseModel):
    summary: str = Field(description="A consolidated summary of all votes.")
    final_choice: str = Field(description="The final, single choice.")

# 2. Define prompt template
VOTE_PROMPT = PromptTemplate(
    "Make a decision based on the following vote results. $format_instructions\n\n##Question:\n$question\n## Vote results:\n$vote_results"
)

# 3. Assemble components
client = get_client(
    endpoint="",
    api_key="",
    model_name="",
)

output_parser = PydanticOutputParser(pydantic_model=FinalDecision)
question = "What is the deepest depth of the world's oceans?"
vote_results = [
    "The Coral Sea is the deepest sea in the world, with a maximum depth of 9,174 meters. It is located in the South Pacific and is famous for its Great Barrier Reef and biodiversity.", 
    "The Milwaukee Deep is located in the Puerto Rico Trench and is 8,380 meters deep, making it the deepest point in the Atlantic Ocean.", 
    "The Fitzer Deep (also known as Challenger Deep) in the Mariana Trench in the Pacific Ocean has a latest measured depth of 11,034 meters."
]

# 4. Run the chain
final_decision, history = run_structured_output_chain(
    client=client,
    prompt_template=VOTE_PROMPT,
    output_parser=output_parser,
    prompt_variables={"question": question, "vote_results": vote_results},
    max_retries=3,
    max_tokens=200,
    temperature=0.0,
)

logging.info(final_decision.model_dump_json(indent=2)) 