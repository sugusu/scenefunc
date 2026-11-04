"""Ordinary category-first retrieval; no model, scene IDs or preferred layouts.

This is a common-backend correction, not a proposed research mechanism. Category
aliases are declared vocabulary, not inferred human preference. Unresolved types
abstain. Detailed style, material and functionality remain unverified.
"""
import html
import math
import re


ALIASES = {
    'bed': ['double_bed', 'king_bed'],
    'queen bed': ['double_bed'],
    'office chair': ['swivel_chair'],
    'dining chair': ['straight_chair'],
    'dresser': ['chest_of_drawers'],
    'side table': ['side_table', 'end_table'],
}


def words(text):
    return re.findall(r'[a-z]+', html.unescape(text).lower().replace('_', ' '))


def label(category):
    return category.split('.')[0]


def resolve_type(short_name, catalog):
    """Longest type suffix, exact category or declared conventional synonym."""
    vocabulary = {label(c).replace('_', ' '): label(c)
                  for row in catalog for c in row['categories']}
    parts = words(short_name)
    for start in range(len(parts)):
        phrase = ' '.join(parts[start:])
        if phrase in ALIASES:
            return {'phrase': phrase, 'categories': ALIASES[phrase],
                    'basis': 'declared_vocabulary_alias'}
        if phrase in vocabulary:
            return {'phrase': phrase, 'categories': [vocabulary[phrase]],
                    'basis': 'exact_catalog_category'}
    return {'phrase': ' '.join(parts), 'categories': [],
            'basis': 'unresolved_type_abstain'}


def rank(description, short_name, desired, catalog, measure):
    """Measure provides frame-normalized real dimensions, or raises a failure.

    Requested dimensions guide ranking only. No hard tolerance, resizing,
    category substitution, task-specific ID or semantic-success declaration.
    """
    if len(desired) != 3 or any(not math.isfinite(v) or v <= 0 for v in desired):
        raise ValueError('Expected three positive finite requested dimensions')
    resolved = resolve_type(short_name, catalog)
    wanted = set(resolved['categories'])
    candidates, errors = [], []
    query = set(words(description))
    for row in catalog:
        if not wanted.intersection(label(c) for c in row['categories']):
            continue
        try:
            measured = measure(row)
            dimensions = measured['dimensions']
            if len(dimensions) != 3 or any(not math.isfinite(x) or x <= 0 for x in dimensions):
                raise ValueError('Invalid mesh dimensions')
            size_error = sum(abs(math.log(a/b)) for a,b in zip(dimensions, desired))
            overlap = len(query.intersection(words(row['name'])))
            candidates.append({'row': row, 'measurement': measured,
                               'dimension_log_error': size_error,
                               'description_overlap': overlap})
        except Exception as exc:
            errors.append({'asset_id': row['id'], 'error': f'{type(exc).__name__}: {exc}'})
    candidates.sort(key=lambda r: (r['dimension_log_error'],
                                   -r['description_overlap'], r['row']['id']))
    return {'resolved_type': resolved, 'candidates': candidates,
            'measurement_failures': errors,
            'success': bool(candidates),
            'failure_reason': None if candidates else
                'No eligible cached mesh in the resolved category; no substitute fabricated',
            'semantic_details_verified': False, 'rescaled': False}
