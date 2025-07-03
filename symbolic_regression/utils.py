# Miscellaneous utility functions (e.g., string similarity, etc.)
# (Move all _string_similarity and similar helpers here)

def string_similarity(s1, s2):
    """Calculate string similarity using character-based Jaccard index"""
    if s1 == s2:
        return 1.0
    set1 = set(s1.replace(' ', '').replace('(', '').replace(')', ''))
    set2 = set(s2.replace(' ', '').replace('(', '').replace(')', ''))
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# Add all other utility functions here.
