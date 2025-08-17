import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.agents.transformation import TransformationAgent
from weaviate.agents.classes import Operations
from weaviate.classes.config import DataType

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY")),
)

collection = client.collections.get("EnronEmails")

add_summary = Operations.update_property(
    property_name="email_summary",
    view_properties=["email_body"],
    instruction="""Your task is to summarize the information contained in this email.
    Please be careful not to miss any important information. It is very important that your summary is accurate and factual.
    """
)

agent = TransformationAgent(
    client=client,
    collection="EnronEmails",
    operations=[add_summary],
)

response = agent.update_all()

print(agent.get_status(workflow_id=response.workflow_id))