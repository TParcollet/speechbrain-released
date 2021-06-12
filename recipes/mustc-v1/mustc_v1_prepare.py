"""
Data preparation.

Download: https://ict.fbk.eu/must-c-releases/

Author
------
Titouan Parcollet
"""

import os
import json
import re
import string
import logging
import unicodedata
from tqdm.contrib import tzip

logger = logging.getLogger(__name__)


def prepare_mustc_v1(
    data_folder,
    save_folder,
    font_case="lc",
    accented_letters=True,
    punctuation=False,
    non_verbal=False,
    tgt_language="de",
    skip_prep=False,
):
    """
    Prepares the json files for the Must-C (V1) Corpus.
    Download: https://ict.fbk.eu/must-c/

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Mustc-C dataset is stored.
        This path should include the lang: https://ict.fbk.eu/must-c-releases/
    save_folder : str
        The directory where to store the json files.
    font_case : str, optional
        Can be tc, lc, uc for True Case, Low Case or Upper Case respectively.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    punctuation : bool, optional
        If set to True, the punctation will be removed.
    non_verbal : bool, optional
        If set to True, non-verbal tags will be removed e.g. ( Applause ).
    tgt_language : str, optional
        Can be "de", "en", "fr", "es", "it", "nl", "pt", "ro" or "ru".
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.mustc-v1.mustc_v1_prepare import prepare_mustc_v1
    >>> data_folder = '/datasets/mustc-v1/en-de/data'
    >>> save_folder = 'exp/mustc'
    >>> prepare_mustc_v1( \
                 data_folder, \
                 save_folder, \
                 tgt_language="de" \
                 )
    """

    if skip_prep or skip(save_folder, tgt_language):
        logger.info("Skipping data preparation.")
        return
    else:
        logger.info("Data preparation ...")

    mustc_v1_languages = ["de", "en", "fr", "es", "it", "nl", "pt", "ro", "ru"]

    if tgt_language not in mustc_v1_languages:
        msg = "tgt_language must be one of:" + str(mustc_v1_languages)
        raise ValueError(msg)

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Additional checks to make sure the data folder contains Common Voice
    check_mustc_folders(data_folder)

    # Setting the official must-c file paths
    train_yaml = os.path.join(data_folder, "train/txt/train.yaml")
    train_src = os.path.join(data_folder, "train/txt/train.en")
    train_tgt = os.path.join(
        data_folder, "train/txt/train." + str(tgt_language)
    )
    train_wav = os.path.join(data_folder, "train/wav")

    dev_yaml = os.path.join(data_folder, "dev/txt/dev.yaml")
    dev_src = os.path.join(data_folder, "dev/txt/dev.en")
    dev_tgt = os.path.join(data_folder, "dev/txt/dev." + str(tgt_language))
    dev_wav = os.path.join(data_folder, "dev/wav")

    test_he_yaml = os.path.join(data_folder, "tst-HE/txt/tst-HE.yaml")
    test_he_src = os.path.join(data_folder, "tst-HE/txt/tst-HE.en")
    test_he_tgt = os.path.join(
        data_folder, "tst-HE/txt/tst-HE." + str(tgt_language)
    )
    test_he_wav = os.path.join(data_folder, "tst-HE/wav")

    test_com_yaml = os.path.join(data_folder, "tst-COMMON/txt/tst-COMMON.yaml")
    test_com_src = os.path.join(data_folder, "tst-COMMON/txt/tst-COMMON.en")
    test_com_tgt = os.path.join(
        data_folder, "tst-COMMON/txt/tst-COMMON." + str(tgt_language)
    )
    test_com_wav = os.path.join(data_folder, "tst-COMMON/wav")

    # Path for preparated csv files
    train = os.path.join(save_folder, "train_en-" + str(tgt_language) + ".json")
    dev = os.path.join(save_folder, "dev_en-" + str(tgt_language) + ".json")
    test_he = os.path.join(
        save_folder, "test_he_en-" + str(tgt_language) + ".json"
    )
    test_com = os.path.join(
        save_folder, "test_com_en-" + str(tgt_language) + ".json"
    )

    # Creating csv files
    create_json(
        train,
        train_yaml,
        train_src,
        train_tgt,
        train_wav,
        data_folder,
        font_case,
        accented_letters,
        punctuation,
        non_verbal,
        tgt_language,
    )

    create_json(
        dev,
        dev_yaml,
        dev_src,
        dev_tgt,
        dev_wav,
        data_folder,
        font_case,
        accented_letters,
        punctuation,
        non_verbal,
        tgt_language,
    )

    create_json(
        test_he,
        test_he_yaml,
        test_he_src,
        test_he_tgt,
        test_he_wav,
        data_folder,
        font_case,
        accented_letters,
        punctuation,
        non_verbal,
        tgt_language,
    )

    create_json(
        test_com,
        test_com_yaml,
        test_com_src,
        test_com_tgt,
        test_com_wav,
        data_folder,
        font_case,
        accented_letters,
        punctuation,
        non_verbal,
        tgt_language,
    )


def skip(save_folder, tgt_language):
    """
    Detects if the must-C data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # File that should exists if already created
    train = os.path.join(save_folder, "train_en-" + str(tgt_language) + ".json")
    dev = os.path.join(save_folder, "dev_en-" + str(tgt_language) + ".json")
    test_he = os.path.join(
        save_folder, "test_he_en-" + str(tgt_language) + ".json"
    )
    test_com = os.path.join(
        save_folder, "test_com_en-" + str(tgt_language) + ".json"
    )

    skip = False

    if (
        os.path.isfile(train)
        and os.path.isfile(dev)
        and os.path.isfile(test_he)
        and os.path.isfile(test_com)
    ):
        skip = True

    return skip


def create_json(
    json_path,
    yaml_path,
    src_path,
    tgt_path,
    wav_path,
    data_folder,
    font_case,
    accented_letters,
    punctuation,
    non_verbal,
    tgt_language,
):
    """
    Creates the a json file.

    See the prepare_mustc_v1 function for the arguments definition.
    """

    # We load all files and check that the number of samples correspond
    loaded_yaml = open(yaml_path, "r").readlines()
    loaded_src = open(src_path, "r").readlines()
    loaded_tgt = open(tgt_path, "r").readlines()

    if not (len(loaded_yaml) == len(loaded_src) == len(loaded_tgt)):
        msg = "The number of lines in yaml, src and tgt files are different!"
        raise ValueError(msg)

    nb_samples = len(loaded_yaml)

    # Adding some Prints
    msg = "Creating json lists in %s ..." % (json_path)
    logger.info(msg)

    msg = "Preparing JSON files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Start processing lines
    total_duration = 0.0
    sample = 0
    cpt = 0

    json_dict = {}

    for line in tzip(loaded_yaml):

        line_yaml = line[0]
        src_trs = loaded_src[cpt]
        tgt_trs = loaded_tgt[cpt]
        cpt += 1

        yaml_split = line_yaml.split(" ")

        # Extract the needed fields
        wav = os.path.join(wav_path, yaml_split[-1].split("}")[0])
        spk_id = yaml_split[-2].split(",")[0]
        snt_id = sample
        duration = float(yaml_split[2].split(",")[0])
        offset = float(yaml_split[4].split(",")[0])
        total_duration += float(duration)

        # Getting transcripts and normalize according to parameters
        normalized_src = src_trs.rstrip()
        normalized_tgt = tgt_trs.rstrip()

        # 1. Case
        if font_case == "lc":
            normalized_src = normalized_src.lower()
            normalized_tgt = normalized_tgt.lower()
        elif font_case == "uc":
            normalized_src = normalized_src.upper()
            normalized_tgt = normalized_tgt.upper()

        # 2. Replace contraction with space
        normalized_src = normalized_src.replace("'", " '")
        normalized_tgt = normalized_tgt.replace("'", " '")

        # 3. Non verbal
        if not non_verbal:
            normalized_src = re.sub(r"\([^()]*\)", "", normalized_src)
            normalized_tgt = re.sub(r"\([^()]*\)", "", normalized_tgt)

        # 4. Punctuation
        if not punctuation:
            normalized_src = normalized_src.translate(
                str.maketrans("", "", string.punctuation)
            )
            normalized_tgt = normalized_tgt.translate(
                str.maketrans("", "", string.punctuation)
            )

        # 5. Accented letters
        if not accented_letters:
            normalized_src = strip_accents(normalized_src)
            normalized_tgt = strip_accents(normalized_tgt)

        # 6. We remove all examples that do not contains anything
        if len(normalized_tgt) < 1 or len(normalized_src) < 1:
            continue

        json_dict[snt_id] = {
            "duration": duration,
            "offset": offset,
            "wav": wav,
            "spk_id": spk_id,
            "wrd_src": normalized_src,
            "wrd_tgt": normalized_tgt,
        }

        sample += 1

    # Writing the dictionary to the json file
    with open(json_path, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    # Final prints
    msg = "%s successfully created!" % (json_path)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(sample))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def check_mustc_folders(data_folder):
    """
    Check if the data folder actually contains the must-c dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain must-c
    """

    train = "/train"

    # Checking clips
    if not os.path.exists(data_folder + train):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the must-c)" % (data_folder + train)
        )
        raise FileNotFoundError(err_msg)


def strip_accents(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)
