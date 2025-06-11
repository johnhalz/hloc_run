import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, field_serializer
from scipy.spatial.transform import Rotation as Rotation

from hloc_run import IMAGE_FORMATS


class Pose(BaseModel):
    translation: np.ndarray  # [x, y, z]
    rotation: Rotation

    class Config:
        arbitrary_types_allowed = True

    @field_serializer("rotation")
    def serialize_rotation(self, value: Rotation) -> list[float]:
        """Serialize rotation to quaternion format"""
        return value.as_matrix().tolist()

    @field_serializer("translation")
    def serialize_translation(self, value: np.ndarray) -> list[float]:
        """Serialize translation to list format"""
        return value.tolist()


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    ox: float
    oy: float

    @property
    def matrix(self) -> np.ndarray:
        """Return camera intrinsics matrix"""
        return np.array([[self.fx, 0, self.ox], [0, self.fy, self.oy], [0, 0, 1]])


class LocalizedImage(BaseModel):
    image_path: Path
    camera_pose: Pose
    camera_intrinsics: CameraIntrinsics

    @field_serializer("image_path")
    def serialize_image_path(self, value: Path) -> str:
        """Serialize image path to string"""
        return value.as_posix()

    @classmethod
    def from_json(cls, image_file: Path, json_file: Path) -> "LocalizedImage":
        """Create LocalizedImage from JSON data"""

        json_data = json.load(json_file.open("r"))

        camera_instrinsics = CameraIntrinsics(
            fx=json_data["fx"],
            fy=json_data["fy"],
            ox=json_data["ox"],
            oy=json_data["oy"],
        )

        camera_pose = Pose(
            translation=np.array([json_data["px"], json_data["py"], json_data["pz"]]),
            rotation=Rotation.from_matrix(
                [
                    [json_data["r00"], json_data["r01"], json_data["r02"]],
                    [json_data["r10"], json_data["r11"], json_data["r12"]],
                    [json_data["r20"], json_data["r21"], json_data["r22"]],
                ]
            ),
        )

        return cls(
            image_path=image_file,
            camera_pose=camera_pose,
            camera_intrinsics=camera_instrinsics,
        )


def get_image_files(folder: Path) -> list[Path]:
    """
    Get all image files in the specified folder.

    Args:
        folder (Path): Path to the folder containing images.

    Returns:
        list[Path]: List of image file paths.
    """
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"The provided path {folder} is not a valid directory.")

    image_files = list(folder.glob("*"))  # Get all files in the folder
    image_files = [f for f in image_files if f.suffix in IMAGE_FORMATS]

    if not image_files:
        raise ValueError(f"No images found in the folder {folder}")

    return image_files
