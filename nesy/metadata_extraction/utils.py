def object_recursive_delete_fields(json_object, fields_to_remove) -> None:
    """
    Recursively delete fields from a JSON object. Nested fields should adhere to structure:
    "field.subfield.subfield.subfield..."
    Args:
        json_object: object read from JSON file from which to remove the fields.
        fields_to_remove: list of fields to remove.
    """
    if isinstance(json_object, list):
        for item in json_object:
            object_recursive_delete_fields(item, fields_to_remove)
    elif isinstance(json_object, dict):
        for field in fields_to_remove:
            parts = field.split(".")
            if len(parts) == 1:
                json_object.pop(parts[0], None)
            else:
                if parts[0] in json_object:
                    object_recursive_delete_fields(
                        json_object[parts[0]], [".".join(parts[1:])]
                    )
        # Recursively apply to nested dictionaries
        for key in list(json_object.keys()):
            object_recursive_delete_fields(json_object[key], fields_to_remove)
