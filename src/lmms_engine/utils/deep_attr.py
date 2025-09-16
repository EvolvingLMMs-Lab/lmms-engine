def has_nested_attr(obj, attr_path: str) -> bool:
    attrs = attr_path.split(".")
    current = obj
    for attr in attrs:
        if not hasattr(current, attr):
            return False
        current = getattr(current, attr)
    return True

def deep_getattr(obj, attr_path: str, default=None):
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default