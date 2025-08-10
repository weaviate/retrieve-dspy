import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.agents.transformation import TransformationAgent
from weaviate.agents.classes import Operations
from weaviate.classes.config import Property, DataType, Configure


client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY")),
)

collection = client.collections.get("FreshstackLangchain")
collection.config.add_property(
    Property(name="summary", data_type=DataType.TEXT)
)
collection.config.add_vector(
    vector_config=Configure.Vectors.text2vec_cohere(
        name="summary_vector",
        source_properties=["summary"]
    )
)

add_summary = Operations.update_property(
    property_name="summary",
    data_type=DataType.TEXT,
    view_properties=["docs_text"],
    instruction="""Create a summary of the document.
    The summary should be a single sentence.
    The summary should be a single sentence."""
)

agent = TransformationAgent(
    client=client,
    collection="FreshstackLangchain",
    operations=[add_summary],
)

response = agent.update_all()

agent.get_status(workflow_id=response.workflow_id)