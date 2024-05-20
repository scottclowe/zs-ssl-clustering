"""
UTKFace dataset.
"""

import datetime
import os
import tarfile

import numpy as np
import pandas as pd
import PIL
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision.datasets.vision import VisionDataset


class UTKFace(VisionDataset):
    """`UTKFace <https://susanqq.github.io/UTKFace/>`_ Dataset.

    Parameters
    ----------
    root : str
        The root directory, to contain the downloaded tarball file, and
        the image directory, UTKFace.
    split : str, default="train"
        The dataset partition, one of ``train``, ``test``, or ``all``.
    target_type : str, default="age"
        Type of target to use, ``age``, ``gender``, or ``race``.
        Can also be a list to output a tuple with all specified target types.
        The targets represent:

            - ``age`` (int): labels (1 to 116) for age of each person
            - ``gender`` (int): binary (0, 1) labels for Male/Female
            - ``race`` (int): labels (0 to 4) representing
              White, Black, Asian, Indian, or Others, respectively.

        Defaults to ``age``. If empty, ``None`` will be returned as target.

    transform : Callable, default=None
        Image transformation pipeline.
    target_transform : Callable, default=None
        Label transformation pipeline.
    download : bool, default=False
        If True, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    """

    base_folder = "UTKFace"
    GENDER_DICT = {0: "Male", 1: "Female"}
    RACE_DICT = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
    EXPECTED_SAMPLE_COUNT = 23_705

    def __init__(
        self,
        root,
        split="train",
        target_type="age",
        transform=None,
        target_transform=None,
        download=False,
    ) -> None:
        root = os.path.expanduser(root)
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Constants
        self.DATA_URL = "https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk"
        self.DATA_FILE_ID = "0BxYys69jI14kYVM3aVhKS1VhRUk"
        self.DATA_ARCHIVE_NAME = "UTKFace.tar.gz"

        # These images do not have all the metadata in their names. They are removed from the dataset.
        self.CORRUPTED_SAMPLES = [
            "39_1_20170116174525125.jpg.chip.jpg",
            "61_1_20170109142408075.jpg.chip.jpg",
            "61_1_20170109150557335.jpg.chip.jpg",
        ]

        self.metadata = None
        self.root = root
        self.image_dir = os.path.expanduser(os.path.join(self.root, self.base_folder))
        self.download = download

        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if not self._check_exists() and download:
            self._download_and_extract()

        if not self._check_exists():
            raise EnvironmentError(
                f"{type(self).__name__} dataset not found in {self.image_dir}."
                " You can use download=True to download it."
            )

        self._extract_metadata()
        assert len(self.metadata) == self.EXPECTED_SAMPLE_COUNT
        self._partition(self.split)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        sample = self.metadata.iloc[index]
        img_path = os.path.join(self.root, self.base_folder, sample["image_name"])
        X = PIL.Image.open(img_path)

        target = []
        for t in self.target_type:
            # TODO
            if t == "age":
                target.append(sample["age"])
            elif t == "gender":
                target.append(sample["gender_id"])
            elif t == "race":
                target.append(sample["race_id"])
            else:
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def _check_exists(self) -> bool:
        """Check if the dataset is already downloaded and extracted.

        Returns
        -------
        bool
            True if the dataset is already downloaded and extracted, False otherwise.
        """
        check = os.path.exists(
            os.path.join(self.image_dir, "1_0_0_20161219140623097.jpg.chip.jpg")
        )
        check &= os.path.exists(
            os.path.join(self.image_dir, "1_0_0_20161219140627985.jpg.chip.jpg")
        )
        check &= os.path.exists(
            os.path.join(self.image_dir, "100_1_2_20170112222336458.jpg.chip.jpg")
        )
        return check

    def _download_and_extract(self) -> None:
        """Download and extract UTKFace dataset.

        Returns
        -------
        None
        """
        os.makedirs(self.image_dir, exist_ok=True)
        print(f'Downloading "{self.DATA_ARCHIVE_NAME}" from Google Drive ...')
        download_file_from_google_drive(
            file_id=self.DATA_FILE_ID, root=self.root, filename=self.DATA_ARCHIVE_NAME
        )
        self._extract_archive()

    def _extract_archive(self) -> None:
        """Extract UTKFace dataset tar archive.

        Returns
        -------
        None
        """
        print(f'Extracting "{self.DATA_ARCHIVE_NAME}" ...')
        with tarfile.open(os.path.join(self.root, self.DATA_ARCHIVE_NAME)) as tar:
            members = tar.getmembers()
            for member in members:
                member.name = os.path.basename(member.name)
                if os.path.splitext(member.name)[1] == "":
                    # Skip directories
                    continue
                tar.extract(member, path=self.image_dir)

    def _extract_metadata(self) -> None:
        """Extract metadata from sample names and creates a pandas DataFrame.

        This function goes through images and extracts age, gender, race, and the date and time of the photos from
        the file names.

        Returns
        -------
        None

        """
        self.image_list = sorted(os.listdir(self.image_dir))

        images_names = []
        images_ages = []
        images_races = []
        images_genders = []
        images_dt = []

        for image in self.image_list:
            if image in self.CORRUPTED_SAMPLES:
                continue
            sample_metadata = image.split(".")[0].strip().split("_")
            image_name = image
            age = sample_metadata[0]
            gender_id = sample_metadata[1]
            race_id = sample_metadata[2]
            dtstr = sample_metadata[3][:17]
            dt = datetime.datetime.strptime(dtstr, "%Y%m%d%H%M%S%f")
            images_names.append(image_name)
            images_ages.append(age)
            images_genders.append(gender_id)
            images_races.append(race_id)
            images_dt.append(dt)

        self.metadata = pd.DataFrame(
            {
                "image_name": images_names,
                "age": images_ages,
                "gender_id": images_genders,
                "race_id": images_races,
                "datetime": images_dt,
            }
        )

        self.metadata["age"] = self.metadata.age.astype("int")
        self.metadata["gender_id"] = self.metadata.gender_id.astype("int")
        self.metadata["race_id"] = self.metadata.race_id.astype("int")
        self.metadata["gender"] = self.metadata["gender_id"].apply(
            lambda x: self.GENDER_DICT[x]
        )
        self.metadata["race"] = self.metadata["race_id"].apply(
            lambda x: self.RACE_DICT[x]
        )

        self.metadata.sort_values(by="age", inplace=True)
        self.metadata.reset_index(drop=True, inplace=True)

        # Add our own partitioning to the data.
        # Select every n-th sample for the test set. Because we sorted the data
        # alphabetically by filename, and then by age, this will be a stratified
        # split not just by age but also by gender and race as much as possible
        # as a sub-stratification of age.
        split_rate = 0.25
        n_test = int(split_rate * len(self.metadata))
        test_indices = np.linspace(
            int(0.5 / split_rate),
            len(self.metadata) - 1 - int(0.5 / split_rate),
            n_test,
        )
        test_indices = np.round(test_indices)
        self.metadata["is_test"] = False
        self.metadata.loc[test_indices, "is_test"] = True
        # Ensure there is at least one sample per age in the train set
        # N.B. A few ages only occur once, so these won't be in the test set.
        sdf = self.metadata[~self.metadata["is_test"]]
        for age in self.metadata["age"].unique():
            if any(sdf["age"] == age):
                continue
            idx = self.metadata.index[self.metadata["age"] == age][0]
            self.metadata.loc[idx, "is_test"] = False

    def _partition(self, split):
        if split == "all":
            pass
        elif split == "train":
            self.metadata = self.metadata[~self.metadata["is_test"]]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["is_test"]]
        else:
            raise ValueError(f"Unknown split {split}")
