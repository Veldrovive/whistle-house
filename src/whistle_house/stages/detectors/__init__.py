"""Whistle detector implementations. Import new modules here to register them.

``learned`` only imports torch when instantiated, so registering it is free.
"""

from whistle_house.stages.detectors import learned, peak_tracker  # noqa: F401
