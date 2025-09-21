from .framework import (
    build_extractors,
    load_extractor,
    register_extractor,
    dedupe_list,
)

# Ensure bundled extractors register themselves
from . import cresskill  # noqa: F401
