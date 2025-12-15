from pathlib import Path

import pandas as pd

import netshare.ray as ray
from netshare import Generator
from netshare.pre_post_processors.netshare.util import csv2pcap_single


def export_best_csv_to_pcap(work_folder: Path, output_pcap: Path) -> None:
    post_dir = work_folder / "post_processed_data"
    csv_candidates = sorted(post_dir.glob("id-*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No post_processed CSVs found in {post_dir}")
    best_csv = csv_candidates[0]
    df = pd.read_csv(best_csv)
    output_pcap.parent.mkdir(parents=True, exist_ok=True)
    csv2pcap_single(df, str(output_pcap))
    print(f"[NetShare] Wrote synthetic PCAP to {output_pcap}")


if __name__ == "__main__":
    ray.config.enabled = False
    ray.init(address="auto")

    config_path = Path(__file__).with_name("config_train_pcap.json")
    work_folder = Path(__file__).resolve().parents[2] / "results" / "train_pcap"
    work_folder.parent.mkdir(parents=True, exist_ok=True)

    generator = Generator(config=str(config_path))
    generator.train(work_folder=str(work_folder))
    generator.generate(work_folder=str(work_folder))
    generator.visualize(work_folder=str(work_folder))

    datasets_dir = Path(__file__).resolve().parents[4] / "datasets"
    export_best_csv_to_pcap(work_folder, datasets_dir / "netshare.pcap")

    ray.shutdown()
