import sys
from pathlib import Path

YOLO_V5_ROOT = Path("/Users/sakith/UCSC-Prac/Research/learning_tasks/yolov5_adversarial")
sys.path.insert(0, str(YOLO_V5_ROOT))

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from detect import run


def main():
	workspace_root = THIS_DIR
	image_path = workspace_root / "data" / "clean" / "images" / "cd6af452-e54a1e36.jpg"
	save_dir = workspace_root / "runs" / "eigen_cam"

	run(
		weights=YOLO_V5_ROOT / "yolov5s.pt",
		source=image_path,
		project=save_dir.parent,
		name=save_dir.name,
		exist_ok=True,
		save_cam=True,
		cam_layer_index=-2,
		cam_task="od",
	)
	print(f"Saved CAM outputs to {save_dir}")


if __name__ == "__main__":
	main()

