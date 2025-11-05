from retrieve_dspy.retrievers import HyDE_QueryExpander
from retrieve_dspy.clients import get_weaviate_client

weaviate_client = get_weaviate_client()

query_expander = HyDE_QueryExpander(
    collection_name="BrightBiology_Default",
    target_property_name="content",
    verbose=True,
    retrieved_k=10,
)

response = query_expander.forward(
    "How many cells are in the human body?", 
    weaviate_client=weaviate_client
)

print(response)