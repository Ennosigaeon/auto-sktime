import argparse
import re
from typing import List


def fold_type(arg_value: str, pat=re.compile(r'^(\*|\d+|\d+-\d+)$')):
    if not pat.match(arg_value):
        raise argparse.ArgumentTypeError(f'Expected single number, range of numbers or "*", got {arg_value}')
    return arg_value


def parse_folds(folds: str, max_folds: int) -> List[int]:
    if folds == '*':
        return list(range(max_folds))
    elif folds.isdigit():
        return [int(folds)]
    elif re.match(r'\d+-\d+', folds):
        tokens = folds.split('-')
        return list(range(int(tokens[0]), int(tokens[1])))
    else:
        raise ValueError()
