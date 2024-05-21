"""
BreakHis dataset.
"""

import os

import PIL
from torchvision.datasets import VisionDataset


class BreakHis(VisionDataset):
    """
    `BreakHis <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>`_ Dataset.

    Parameters
    ----------
    root : string
        Root directory of the dataset.
    train : bool, optional
        If True, creates dataset from training set, otherwise creates from
        test set.
    target_type : str, optional
        Type of target to use. One of ``"malignancy"`` (binary tumor class),
        ``tumor`` (tumor type), ``slide`` (slide id), ``magnification``, or
        ``tumor-magnification`` (a class for each pair of tumor type and
        magnification).
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a
        transformed version. e.g, ``transforms.RandomCrop``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    """

    base_folder = os.path.join("histology_slides", "breast")
    EXPECTED_SAMPLE_COUNT = 7_909
    labels_malignancy = ["B", "M"]
    labels_tumortype = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
    labels_magnification = [40, 100, 200, 400]
    labels_slideid = [
        "10147",
        "10926",
        "11031",
        "11520",
        "11951",
        "12204",
        "12312",
        "12465",
        "12773",
        "13200",
        "13412",
        "13413",
        "13418DE",
        "13993",
        "14015",
        "14134",
        "14134E",
        "14926",
        "14946",
        "15275",
        "15570",
        "15570C",
        "15572",
        "15687B",
        "15696",
        "15704",
        "15792",
        "16184",
        "16184CD",
        "16188",
        "16196",
        "16336",
        "16448",
        "16456",
        "16601",
        "16716",
        "16875",
        "17614",
        "17901",
        "17915",
        "18650",
        "18842",
        "18842D",
        "190EF",
        "19440",
        "19854C",
        "19979",
        "19979C",
        "20629",
        "20636",
        "21978AB",
        "21998AB",
        "21998CD",
        "21998EF",
        "22549AB",
        "22549CD",
        "22549G",
        "22704",
        "23060AB",
        "23060CD",
        "23222AB",
        "25197",
        "2523",
        "2773",
        "29315EF",
        "2980",
        "2985",
        "29960AB",
        "29960CD",
        "3411F",
        "3909",
        "4364",
        "4372",
        "5287",
        "5694",
        "5695",
        "6241",
        "8168",
        "9133",
        "9146",
        "9461",
    ]

    def __init__(
        self,
        root,
        target_type="tumortype",
        transform=None,
        target_transform=None,
    ):
        root = os.path.expanduser(root)
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.image_dir = os.path.join(root, self.base_folder)

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if not self._check_exists():
            raise EnvironmentError(
                f"{type(self).__name__} dataset not found in {self.image_dir}."
            )

        self._build_metadata()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        fname = os.path.join(self.image_dir, self.image_paths[index])
        X = PIL.Image.open(fname)

        target = []
        for t in self.target_type:
            if t in ["tumortype-magnification", "magnification-tumortype"]:
                # Encodes the joint of tumortype and magnification
                target.append(
                    self.magnification_indices[index] * len(self.labels_tumortype)
                    + self.tumortype_indices[index]
                )
            elif t in ["malignant", "malignancy"]:
                target.append(self.malignancy_indices[index])
            elif t in ["tumor", "tumortype", "type"]:
                target.append(self.tumortype_indices[index])
            elif t == "magnification":
                target.append(self.magnification_indices[index])
            elif t in ["slideid", "slide_id"]:
                target.append(self.slideid_indices[index])
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
            os.path.join(
                self.image_dir,
                "benign/SOB/tubular_adenoma/SOB_B_TA_14-21978AB/400X/SOB_B_TA-14-21978AB-400-013.png",
            )
        )
        check &= os.path.exists(
            os.path.join(
                self.image_dir,
                "malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-19979/100X/SOB_M_MC-14-19979-100-017.png",
            )
        )
        return check

    def _build_metadata(self):
        image_paths = [
            os.path.join(subdir, file)[len(self.image_dir) + 1 :]
            for subdir, _, files in os.walk(self.image_dir)
            for file in files
            if file.startswith("SOB_") and file.endswith(".png")
        ]
        # Check we have found all the images
        assert len(image_paths) == self.EXPECTED_SAMPLE_COUNT
        # Sort alphabetically so the order is consistent
        image_paths = sorted(image_paths)
        self.image_paths = image_paths
        # Get metadata from filenames
        malignancies = []
        tumor_types = []
        magnifications = []
        slideids = []
        for image_path in image_paths:
            fname = os.path.basename(image_path)
            # Malignant or Benign
            malignancy = fname[4]
            if malignancy not in ["M", "B"]:
                raise ValueError()
            # Tumor type
            parts = fname.split("_")
            parts2 = parts[-1].split("-")
            tumor_type = parts2[0]
            # Magnification level
            magnification = int(parts2[3])
            # Slide ID
            slideid = parts2[2]
            # Add to lists
            malignancies.append(malignancy)
            tumor_types.append(tumor_type)
            magnifications.append(magnification)
            slideids.append(slideid)
        self.malignancies = malignancies
        self.tumortypes = tumor_types
        self.magnifications = magnifications
        self.slideids = slideids
        # Convert metadata to indices
        self.malignancy_indices = [
            self.labels_malignancy.index(k) for k in self.malignancies
        ]
        self.tumortype_indices = [
            self.labels_tumortype.index(k) for k in self.tumortypes
        ]
        self.magnification_indices = [
            self.labels_magnification.index(k) for k in self.magnifications
        ]
        self.slideid_indices = [self.labels_slideid.index(k) for k in self.slideids]
