from typing import Union


def check_true(p: Union[str, bool, int]) -> bool:
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p: Union[str, bool, int]) -> bool:
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p: Union[str, bool, int, float]) -> bool:
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p: Union[str, bool, int]) -> bool:
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError(f'{p} is not a bool')
