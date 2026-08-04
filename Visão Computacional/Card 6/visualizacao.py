from pathlib import Path

import fiftyone as fo


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	dataset_dir = base_dir / "dataset_anotado"

	dataset = fo.Dataset.from_dir(
		dataset_dir=str(dataset_dir),
		dataset_type=fo.types.COCODetectionDataset,
		data_path="images/default",
		labels_path="annotations/instances_default.json",
		label_field="ground_truth",
		name="dataset_anotado",
		overwrite=True,
	)

	fo.launch_app(dataset).wait()


if __name__ == "__main__":
	main()
