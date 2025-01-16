""" Simple utilities to load the mysterious large scale ASR dataset from HuggingFace. This does not actually prepare the large scale ASR dataset. For this, please refer to the Preparation folder.

Author
-------
 * Titouan Parcollet, 2024
"""

import multiprocessing
import os

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def load_datasets(subset, hf_download_folder, hf_caching_dir):
    """Load and create the HuggingFace dataset for the large scale ASR. It must
    have been downloaded manually into hf_download_folder first. This function
    operates in an "offline" mode and will not try to download the dataset.

    Parameters
    ----------
    subset: str
        Name of the subset of interest: one of [large, medium, small, clean]
    hf_download_folder : str
        The path where HF stored the dataset.
    hf_caching_dir : str
        The path where HF will extract (or not if already done) the dataset.

    Returns
    -------
    Dictionary of HuggingFace dataset. ["train", "dev", "test"]
    """

    try:
        import datasets
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            f"{str(error)}\n" f"HuggingFace datasets must be installed."
        )

    if not os.path.isdir(hf_download_folder):
        msg = "You must download the dataset with HuggingFace before starting "
        msg += "this recipe. Please check the HuggingFace hub of The large scale ASR."
        raise ValueError(msg)

    # Managing the download dir as HF can be capricious with this.
    logger.info("Caching to: " + str(hf_caching_dir))

    nproc = multiprocessing.cpu_count()
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    datasets.disable_progress_bars()

    hf_data = load_dataset(
        hf_download_folder,
        name=subset,
        num_proc=nproc,
        cache_dir=hf_caching_dir,
    )

    os.environ["HF_DATASETS_OFFLINE"] = "0"

    return hf_data
