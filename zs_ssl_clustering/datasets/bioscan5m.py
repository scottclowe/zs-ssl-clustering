"""
BIOSCAN-5M dataset.
"""

import os

import pandas as pd
import PIL
from torchvision.datasets.vision import VisionDataset

df_dtypes = {
    "processid": "str",
    "sampleid": "str",
    "image_file": str,
    "chunk_number": "uint8",
    "phylum": "category",
    "class": "category",
    "order": "category",
    "family": "category",
    "subfamily": "category",
    "genus": "category",
    "species": "category",
    "dna_bin": "category",
    "dna_barcode": str,
    "split": "category",
    "country": "category",
    "province/state": "category",
    "coord-lat": float,
    "coord-lon": float,
    "surface_area": float,
    "bioscan1M_index": "Int64",
    "label_was_inferred": "uint8",
}

df_usecols = [
    "image_file",
    "chunk_number",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "genus",
    "species",
    "dna_bin",
    "dna_barcode",
    "split",
]


class BIOSCAN5M(VisionDataset):
    """
    BIOSCAN-5M Dataset.

    Parameters
    ----------
    root : str
        The root directory, to contain the downloaded tarball file, and
        the image directory, BIOSCAN-5M.

    split : str, default="train"
        The dataset partition.

    modality : str or Tuple[str], default=("image", "dna")
        Which data modalities to use.

    reduce_repeated_barcodes : str or bool, default=False
        Whether to reduce the dataset to only one sample per barcodes.

    max_nucleotides : int, default=None
        Maximum number of nucleotides to keep in the DNA barcode.

    target_type : str, default="species"
        Type of target to use. One of:
        ``"phylum"``, ``"class"``, ``"order"``, ``"family"``, ``"subfamily"``,
        ``"genus"``, ``"species"``, ``"dna_bin"``.

    transform : Callable, default=None
        Image transformation pipeline.

    dna_transform : Callable, default=None
        Barcode DNA transformation pipeline.

    target_transform : Callable, default=None
        Label transformation pipeline.
    """

    def __init__(
        self,
        root,
        split="train",
        modality=("image", "dna"),
        reduce_repeated_barcodes=False,
        max_nucleotides=None,
        target_type="species",
        transform=None,
        dna_transform=None,
        target_transform=None,
    ) -> None:
        root = os.path.expanduser(root)
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.metadata = None
        self.root = root
        self.image_dir = os.path.expanduser(os.path.join(self.root, "images"))

        self.split = split
        self.reduce_repeated_barcodes = reduce_repeated_barcodes
        self.max_nucleotides = max_nucleotides
        self.dna_transform = dna_transform

        if isinstance(modality, str):
            self.modality = [modality]
        else:
            self.modality = list(modality)

        if isinstance(target_type, str):
            self.target_type = [target_type]
        else:
            self.target_type = list(target_type)

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if not self._check_exists():
            raise EnvironmentError(
                f"{type(self).__name__} dataset not found in {self.image_dir}."
            )

        self.metadata = self._load_metadata()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        sample = self.metadata.iloc[index]
        img_path = os.path.join(
            self.image_dir,
            f"cropped_resized_part{sample['chunk_number']}",
            f"cropped_resized_{sample['image_file']}",
        )
        values = []
        for modality in self.modality:
            if modality == "image":
                X = PIL.Image.open(img_path)
                if self.transform is not None:
                    X = self.transform(X)
            elif modality == "dna":
                X = sample["dna_barcode"]
                if self.dna_transform is not None:
                    X = self.dna_transform(X)
            else:
                raise ValueError(f"Unfamiliar modality: {modality}")
            values.append(X)

        target = []
        for t in self.target_type:
            target.append(sample[f"{t}_index"])

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        values.append(target)
        return tuple(values)

    def _check_exists(self) -> bool:
        """Check if the dataset is already downloaded and extracted.

        Returns
        -------
        bool
            True if the dataset is already downloaded and extracted, False otherwise.
        """
        check = os.path.exists(os.path.join(self.root, "BIOSCAN-5M_Dataset.csv"))
        if "image" in self.modality:
            # Only check the images exist if the images folder exists,
            # as the user might only be interested in the DNA data
            check &= os.path.exists(
                os.path.join(self.image_dir, "cropped_resized_part1")
            )
            check &= os.path.exists(
                os.path.join(self.image_dir, "cropped_resized_part2")
            )
            check &= os.path.exists(
                os.path.join(self.image_dir, "cropped_resized_part99")
            )
        return check

    def _load_metadata(self) -> pd.DataFrame:
        """Extract metadata from sample names and creates a pandas DataFrame.

        This function goes through images and extracts age, gender, race, and the date and time of the photos from
        the file names.

        Returns
        -------
        None
        """
        df = pd.read_csv(
            os.path.join(self.root, "BIOSCAN-5M_Dataset.csv"),
            dtype=df_dtypes,
            usecols=df_usecols,
        )
        if self.max_nucleotides is not None:
            df["dna_barcode"] = df["dna_barcode"].str[: self.max_nucleotides]
        if self.reduce_repeated_barcodes:
            # Shuffle the data order
            df = df.sample(frac=1, random_state=0)
            # Drop duplicated barcodes
            if self.reduce_repeated_barcodes == "rstrip_Ns":
                df["dna_barcode_strip"] = df["dna_barcode"].str.rstrip("N")
                df = df.drop_duplicates(subset=["dna_barcode_strip"])
                df.drop(columns=["dna_barcode_strip"], inplace=True)
            elif self.reduce_repeated_barcodes == "base":
                df = df.drop_duplicates(subset=["dna_barcode"])
            else:
                raise ValueError(
                    f"Unfamiliar reduce_repeated_barcodes value: {self.reduce_repeated_barcodes}"
                )
            # Re-order the data (reverting the shuffle)
            df = df.sort_index()
        # Filter to just the split of interest
        df = df[df["split"] == self.split]
        # Add index columns to use for targets
        label_cols = [
            "phylum",
            "class",
            "order",
            "family",
            "subfamily",
            "genus",
            "species",
            "dna_bin",
        ]
        for c in label_cols:
            df[c + "_index"] = df[c].cat.codes
        return df
