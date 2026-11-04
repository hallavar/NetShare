import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

import netshare.ray as ray
from netshare import Generator
from netshare.pre_post_processors.netshare.util import csv2pcap_single

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCHMARK_ROOT / "utils"))
from pcap_validation import validate_readable_pcap


def preprocessed_data_ready(generator: Generator, work_folder: Path) -> bool:
    pre_dir = Path(generator._get_pre_processed_data_folder(str(work_folder)))
    if not pre_dir.exists():
        return False
    if not ((pre_dir / "train.csv").exists() or (pre_dir / "raw.csv").exists()):
        return False

    expected_chunks = int(generator._config["global_config"]["n_chunks"])
    required_chunk_files = [
        "data_train.npz",
        "data_attribute_output.pkl",
        "data_feature_output.pkl",
        "raw.csv",
    ]
    for chunk_id in range(expected_chunks):
        chunk_dir = pre_dir / f"chunkid-{chunk_id}"
        if not chunk_dir.is_dir():
            return False
        for filename in required_chunk_files:
            if not (chunk_dir / filename).exists():
                return False
    return True


def trim_pcap_to_duration(output_pcap: Path, reference_pcap: Path, duration: int) -> None:
    trim_script = BENCHMARK_ROOT / "utils" / "trim_pcap.py"
    if not trim_script.exists():
        raise FileNotFoundError(trim_script)
    cmd = [
        sys.executable,
        str(trim_script),
        "--input",
        str(output_pcap),
        "--output",
        str(output_pcap),
        "--duration",
        str(duration),
        "--reference-pcap",
        str(reference_pcap),
    ]
    import subprocess

    subprocess.check_call(cmd)


def estimate_target_packets(reference_pcap: Path, duration: int) -> int:
    target_script = BENCHMARK_ROOT / "utils" / "pcap_target_packets.py"
    if not target_script.exists():
        raise FileNotFoundError(target_script)
    import subprocess

    out = subprocess.check_output(
        [
            sys.executable,
            str(target_script),
            "--pcap",
            str(reference_pcap),
            "--duration",
            str(duration),
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
    if target_packets is not None:
        if df.empty:
            raise RuntimeError(f"Generated CSV is empty: {best_csv}")
        if target_packets > len(df):
            repeats = (target_packets + len(df) - 1) // len(df)
            df = pd.concat([df] * repeats, ignore_index=True)
        df = df.iloc[:target_packets]
    output_pcap.parent.mkdir(parents=True, exist_ok=True)
    csv2pcap_single(df, str(output_pcap))
    validate_readable_pcap(output_pcap)
    print(f"[NetShare] Wrote synthetic PCAP to {output_pcap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NetShare PCAP example pipeline.")
    parser.add_argument(
        "--start-from",
        choices=["train", "generate", "post_process", "visualize", "export"],
        default="train",
        help="Skip earlier stages and resume from this step.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["train", "generate", "post_process", "visualize", "export"],
        default="export",
        help="Stop after completing this step.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip the SDMetrics Dash visualization step (useful on headless machines).",
    )
    parser.add_argument(
        "--reuse-preprocessed",
        action="store_true",
        default=True,
        help="Reuse existing pre_processed_data when it already contains all chunk artifacts.",
    )
    parser.add_argument(
        "--target-packets",
        type=int,
        default=None,
        help="If set, pad the exported pcap up to this many packets; otherwise count from test.pcap and --duration.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=int(os.environ.get("TRAFFIC_DURATION_SECONDS", "600")),
        help="Duration in seconds for target packet estimation and final trimming.",
    )
    args = parser.parse_args()

    steps = ["train", "generate", "post_process", "visualize", "export"]
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
        if args.reuse_preprocessed and preprocessed_data_ready(generator, work_folder):
            print("[NetShare] Reusing existing pre_processed_data and skipping preprocessing")
            generator._train(
                input_train_data_folder=generator._get_pre_processed_data_folder(str(work_folder)),
                output_model_folder=generator._get_model_folder(str(work_folder)),
                log_folder=generator._get_model_log_folder(str(work_folder)),
            )
        else:
            generator.train(work_folder=str(work_folder))
    if start_idx <= steps.index("generate") <= stop_idx:
        if not generator._generate(
            input_train_data_folder=generator._get_pre_processed_data_folder(str(work_folder)),
            input_model_folder=generator._get_model_folder(str(work_folder)),
            output_syn_data_folder=generator._get_generated_data_folder(str(work_folder)),
            log_folder=generator._get_generated_data_log_folder(str(work_folder)),
        ):
            raise RuntimeError("Failed to generate synthetic data")
    if start_idx <= steps.index("post_process") <= stop_idx:
        if not generator._post_process(
            input_folder=generator._get_generated_data_folder(str(work_folder)),
            output_folder=generator._get_post_processed_data_folder(str(work_folder)),
            log_folder=generator._get_post_processed_data_log_folder(str(work_folder)),
            pre_processed_data_folder=generator._get_pre_processed_data_folder(str(work_folder)),
        ):
            raise RuntimeError("Failed to post-process data")
    if (
        not args.skip_visualize
        and start_idx <= steps.index("visualize") <= stop_idx
    ):
        generator.visualize(work_folder=str(work_folder))
    if start_idx <= steps.index("export") <= stop_idx:
        datasets_dir = Path(
            os.environ.get(
                "DATASETS_DIR",
                "/home/aschoen/Programs/dataset_april_imc_paper",
            )
        )
        if args.target_packets is None:
            reference_pcap = datasets_dir / "test.pcap"
            print(
                f"[NetShare] Counting target packets in first {args.duration}s of "
                f"{reference_pcap}"
            )
            target_packets = estimate_target_packets(reference_pcap, args.duration)
            print(f"[NetShare] Target packet count: {target_packets}")
        else:
            target_packets = args.target_packets
            print(f"[NetShare] Using target packet count: {target_packets}")
        output_pcap = datasets_dir / "netshare.pcap"
        export_best_csv_to_pcap(
            work_folder,
            output_pcap,
            target_packets=target_packets,
        )
        trim_pcap_to_duration(output_pcap, datasets_dir / "test.pcap", args.duration)
        validate_readable_pcap(output_pcap)

    ray.shutdown()


if __name__ == "__main__":
    main()
