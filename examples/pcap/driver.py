import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

import netshare.ray as ray
from netshare import Generator
from netshare.pre_post_processors.netshare.util import csv2pcap_single


def trim_pcap_to_minute(output_pcap: Path, reference_pcap: Path) -> None:
    trim_script = Path(__file__).resolve().parents[3] / "utils" / "trim_pcap.py"
    target_script = Path(__file__).resolve().parents[3] / "utils" / "pcap_target_packets.py"
    if not trim_script.exists():
        raise FileNotFoundError(trim_script)
    if not target_script.exists():
        raise FileNotFoundError(target_script)
    cmd = [
        "python3",
        str(trim_script),
        "--input",
        str(output_pcap),
        "--output",
        str(output_pcap),
        "--duration",
        "60",
        "--reference-pcap",
        str(reference_pcap),
    ]
    import subprocess

    subprocess.check_call(cmd)


def estimate_target_packets(train_pcap: Path) -> int:
    target_script = Path(__file__).resolve().parents[3] / "utils" / "pcap_target_packets.py"
    if not target_script.exists():
        raise FileNotFoundError(target_script)
    import subprocess

    out = subprocess.check_output(
        [
            "python3",
            str(target_script),
            "--pcap",
            str(train_pcap),
            "--duration",
            "60",
        ],
        text=True,
    )
    return int(out.strip())


def export_best_csv_to_pcap(
    work_folder: Path,
    output_pcap: Path,
    target_packets: Optional[int] = None,
) -> None:
    post_dir = work_folder / "post_processed_data"
    csv_candidates = sorted(post_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No post_processed CSVs found in {post_dir}")
    best_csv = csv_candidates[0]
    df = pd.read_csv(best_csv)
    if target_packets is not None and target_packets > len(df):
        repeats = (target_packets + len(df) - 1) // len(df)
        df = pd.concat([df] * repeats, ignore_index=True).iloc[:target_packets]
    output_pcap.parent.mkdir(parents=True, exist_ok=True)
    csv2pcap_single(df, str(output_pcap))
    print(f"[NetShare] Wrote synthetic PCAP to {output_pcap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NetShare PCAP example pipeline.")
    parser.add_argument(
        "--start-from",
        choices=["train", "generate", "visualize", "export"],
        default="train",
        help="Skip earlier stages and resume from this step.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["train", "generate", "visualize", "export"],
        default="export",
        help="Stop after completing this step.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip the SDMetrics Dash visualization step (useful on headless machines).",
    )
    parser.add_argument(
        "--target-packets",
        type=int,
        default=1662000,
        help="If set, pad the exported pcap up to this many packets (default matches test.pcap).",
    )
    args = parser.parse_args()

    steps = ["train", "generate", "visualize", "export"]
    start_idx = steps.index(args.start_from)
    stop_idx = steps.index(args.stop_after)
    if start_idx > stop_idx:
        raise ValueError("--start-from cannot come after --stop-after")

    ray.config.enabled = False
    ray.init(address="auto")

    config_path = Path(__file__).with_name("config_train_pcap.json")
    work_folder = Path(__file__).resolve().parents[2] / "results" / "train_pcap"
    work_folder.parent.mkdir(parents=True, exist_ok=True)

    generator = Generator(config=str(config_path))

    if start_idx <= steps.index("train") <= stop_idx:
        generator.train(work_folder=str(work_folder))
    if start_idx <= steps.index("generate") <= stop_idx:
        generator.generate(work_folder=str(work_folder))
    if (
        not args.skip_visualize
        and start_idx <= steps.index("visualize") <= stop_idx
    ):
        generator.visualize(work_folder=str(work_folder))
    if start_idx <= steps.index("export") <= stop_idx:
        datasets_dir = Path(__file__).resolve().parents[4] / "datasets"
        target_packets = estimate_target_packets(datasets_dir / "train.pcap")
        output_pcap = datasets_dir / "netshare.pcap"
        export_best_csv_to_pcap(
            work_folder,
            output_pcap,
            target_packets=target_packets,
        )
        trim_pcap_to_minute(output_pcap, datasets_dir / "test.pcap")

    ray.shutdown()


if __name__ == "__main__":
    main()
