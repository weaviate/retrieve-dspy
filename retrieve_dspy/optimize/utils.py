import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter  # v4 filter helper

def get_content_from_dataset_id(dataset_id: str) -> str:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )  # [[Cloud connect](https://docs.weaviate.io/weaviate/connections/connect-cloud)]

    try:
        collection = client.collections.use("BrightBiology_Default")

        # Filter on the text property "dataset_id"
        result = collection.query.fetch_objects(
            filters=Filter.by_property("dataset_id").equal(dataset_id),  # [[Filters](https://docs.weaviate.io/weaviate/search/filters)]
            limit=1,
        )

        if not result.objects:
            return ""

        return result.objects[0].properties.get("content", "")
    finally:
        client.close()

if __name__ == "__main__":
    print(get_content_from_dataset_id("organism_learn/Learning_14_160.txt"))