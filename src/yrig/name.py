def get_short_name(transform: str) -> str:
    return transform.rsplit("|", 1)[-1]
