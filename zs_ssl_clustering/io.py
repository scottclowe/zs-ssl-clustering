import os
import re


def sanitize_filename(text, allow_dotfiles=False):
    """
    Sanitise string so it can be a filename.

    Parameters
    ----------
    text : str
        A string.
    allow_dotfiles : bool, optional
        Whether to allow leading periods. Leading periods indicate hidden files
        on Linux. Default is `False`.

    Returns
    -------
    text : str
        The sanitised filename string.
    """
    # Remove non-ascii characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Folder names cannot end with a period in Windows. In Unix, a leading
    # period means the file or folder is normally hidden.
    # For this reason, we trim away leading and trailing periods as well as
    # spaces.
    if allow_dotfiles:
        text = text.strip().rstrip(".")
    else:
        text = text.strip(" .")

    # On Windows, a filename cannot contain any of the following characters:
    # \ / : * ? " < > |
    # Other operating systems are more permissive.
    # Replace / with a hyphen
    text = text.replace("/", "-")
    # Use a blacklist to remove any remaining forbidden characters
    text = re.sub(r'[\/:*?"<>|]+', "", text)
    return text


def get_embeddings_path(config):
    """
    Generate path to embeddings file.
    """
    fname = config.dataset_name + "__" + config.model + ".npz"
    fname = sanitize_filename(fname)
    fname = os.path.join(
        config.embedding_dir,
        sanitize_filename(config.partition + f"__z{config.zoom_ratio}"),
        fname,
    )
    return fname
