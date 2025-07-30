import time
import weaviate.collections.classes.config as wvcc

def database_loader(
    weaviate_client,
    dataset_name: str, 
    objects: dict
):
    if dataset_name == "enron":
        if weaviate_client.collections.exists("EnronEmails"):
            weaviate_client.collections.delete("EnronEmails")
        
        weaviate_client.collections.create(
            name="EnronEmails",
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            properties=[
                wvcc.Property(name="email_body", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT),
            ],
        )

        start_time = time.time()
        with weaviate_client.batch.fixed_size(batch_size=100, concurrent_requests=4) as batch:
            for i, email_with_id in enumerate(objects):
                batch.add_object(
                    collection="EnronEmails",
                    properties={
                        "email_body": email_with_id["email_body"],
                        "dataset_id": str(email_with_id["dataset_id"])
                    }
                )
                if i % 1000 == 999:
                    print(f"Inserted {i + 1} emails into Weaviate... (Time elapsed: {time.time()-start_time:.2f} seconds)")

        end_time = time.time()
        upload_time = end_time - start_time
        print(f"Inserted {i + 1} emails into Weaviate... (Time elapsed: {upload_time:.2f} seconds)")
    if dataset_name == "wixqa":
        if weaviate_client.collections.exists("WixKB"):
            weaviate_client.collections.delete("WixKB")
        
        weaviate_client.collections.create(
            name="WixKB",
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            properties=[
                wvcc.Property(name="contents", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="title", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="article_type", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT)
            ]
        )

        start_time = time.time()
        with weaviate_client.batch.fixed_size(batch_size=100, concurrent_requests=4) as batch:
            for i, kb_item_with_id in enumerate(objects):
                batch.add_object(
                    collection="WixKB",
                    properties={
                        "contents": kb_item_with_id["contents"],
                        "title": kb_item_with_id["title"],
                        "article_type": kb_item_with_id["article_type"],
                        "dataset_id": kb_item_with_id["id"]
                    }
                )
                if i % 1000 == 999:
                    print(f"Inserted {i + 1} documents into Weaviate... (Time elapsed: {time.time()-start_time:.2f} seconds)")

            end_time = time.time()
            upload_time = end_time - start_time
            print(f"Inserted {i + 1} documents into Weaviate... (Time elapsed: {upload_time:.2f} seconds)")
        
    if dataset_name.startswith("freshstack-"):
        collection_name = f"Freshstack{dataset_name.split('-')[1].capitalize()}"
        
        if weaviate_client.collections.exists(collection_name):
            weaviate_client.collections.delete(collection_name)
        
        weaviate_client.collections.create(
            name=collection_name,
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec_weaviate(),
            properties=[
                wvcc.Property(name="docs_text", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT),
            ],
        )

        start_time = time.time()
        with weaviate_client.batch.fixed_size(batch_size=100, concurrent_requests=4) as batch:
            for i, doc in enumerate(objects):
                batch.add_object(
                    collection=collection_name,
                    properties={
                        "docs_text": doc["text"],
                        "dataset_id": str(doc["dataset_id"])
                    }
                )
                if i % 1000 == 999:
                    print(f"Inserted {i + 1} documents into Weaviate... (Time elapsed: {time.time()-start_time:.2f} seconds)")

        end_time = time.time()
        upload_time = end_time - start_time
        print(f"Inserted {i + 1} documents into Weaviate... (Time elapsed: {upload_time:.2f} seconds)")