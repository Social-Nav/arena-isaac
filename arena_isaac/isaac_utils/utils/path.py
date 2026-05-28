import os
import re


def sanitize_path_component(component: str) -> str:
    if component and not re.match(r'^[a-zA-Z_]', component):
        return f'_{component}'
    return component


def world_path(*path: str) -> str:
    if len(path) == 1:
        raw_path = path[0]
        normalized = os.path.normpath(raw_path)
        if normalized == '/World' or normalized.startswith('/World' + os.sep):
            return normalized
        path = tuple(raw_path.split(os.sep))
    return os.path.join('/World', *filter(None, map(sanitize_path_component, path)))
