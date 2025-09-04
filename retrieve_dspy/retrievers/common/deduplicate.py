from retrieve_dspy.models import ObjectFromDB

def deduplicate(original_list: list[ObjectFromDB], incoming_list: list[ObjectFromDB]) -> list[ObjectFromDB]:
    seen_ids = set()
    for obj in original_list:
        seen_ids.add(obj.object_id)
    for obj in incoming_list:
        if obj.object_id not in seen_ids:
            seen_ids.add(obj.object_id)
            original_list.append(obj)
    return original_list