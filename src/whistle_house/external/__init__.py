"""External datasets (downloaded on a training machine, never on the Pi).

    whistle-house datasets list
    whistle-house datasets fetch all        # download + prepare into $WHISTLE_HOUSE_EXTERNAL

See base.py for the on-disk format and datasets.py for the built-in datasets.
"""

from whistle_house.external.base import (  # noqa: F401
    ClipEntry,
    ExternalDataset,
    dataset_names,
    dataset_status,
    get_dataset,
    load_manifest,
)
