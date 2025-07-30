import os

import weaviate

from retrieve_dspy.datasets.in_memory import in_memory_dataset_loader
from retrieve_dspy.datasets.populate_db import database_loader

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)

dataset_name = "freshstack-angular"

documents, _ = in_memory_dataset_loader(dataset_name)

database_loader(weaviate_client, dataset_name, documents)

weaviate_client.close()