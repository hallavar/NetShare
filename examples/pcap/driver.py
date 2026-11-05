import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

import netshare.ray as ray
from netshare import Generator
from netshare.pre_post_processors.netshare.util import csv2pcap_single

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCHMARK_ROOT / "utils"))
from pcap_validation import validate_readable_pcap
from pcap_size import generate_to_size, resolve_target
from random_seed import derive_seed, environment_seed, seed_rngs


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


def iter_best_csv_pcaps(
    work_folder: Path,
    directory: Path,
):
    post_dir = work_folder / "post_processed_data"
    csv_candidates = sorted(post_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No post_processed CSVs found in {post_dir}")
    best_csv = csv_candidates[0]
    # choose_best_model writes CSVs sorted by time. Export in bounded pieces so
    # a large reference does not require millions of Scapy packets in memory.
    with pd.read_csv(best_csv, chunksize=10_000) as chunks:
        for index, frame in enumerate(chunks):
            if frame.empty:
                continue
            pcap = directory / f"part-{index:06d}.pcap"
            try:
                csv2pcap_single(frame, str(pcap))
                yield pcap
            finally:
                pcap.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NetShare PCAP example pipeline.")
    parser.add_argument("--seed", type=int, default=environment_seed())
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
        "--target-bytes",
        type=int,
        default=int(os.environ.get("TARGET_BYTES", "0")),
        help="Target PCAP file bytes; 0 matches the entire reference.pcap.",
    )
    parser.add_argument(
        "--reference-pcap",
        type=Path,
        default=None,
        help="Reference capture (defaults to REFERENCE_PCAP or DATASETS_DIR/reference.pcap).",
    )
    parser.add_argument("--max-generation-batches", type=int,
                        default=int(os.environ.get("MAX_GENERATION_BATCHES", "1000")))
    args = parser.parse_args()
    seed_rngs(args.seed, torch=True)

    steps = ["train", "generate", "post_process", "visualize", "export"]
    start_idx = steps.index(args.start_from)
    stop_idx = steps.index(args.stop_after)
    if start_idx > stop_idx:
        raise ValueError("--start-from cannot come after --stop-after")

    datasets_dir = Path(os.environ.get("DATASETS_DIR", "/media/datasets/aschoen_datasets/dataset_campus"))
    reference_pcap = args.reference_pcap or Path(os.environ.get(
        "REFERENCE_PCAP", os.environ.get("TEST_PCAP", str(datasets_dir / "reference.pcap"))))
    target_bytes = resolve_target(reference_pcap, args.target_bytes)
    print(f"[NetShare] Target PCAP size: {target_bytes} bytes")
    config_path = Path(__file__).with_name("config_train_pcap.json")
    run_name = os.environ.get("RUN_NAME", "campus_reference_size")
    work_folder = Path(os.environ.get("NETSHARE_WORK_FOLDER", str(
        Path(__file__).resolve().parents[2] / "results" / run_name)))
    work_folder.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    config["model"]["config"]["seed"] = args.seed
    config["model"]["config"]["generation_seed"] = derive_seed(args.seed, "generation", 1)
    config["global_config"]["original_data_file"] = os.environ.get("TRAIN_PCAP", str(datasets_dir / "train.pcap"))
    # Resolve the input before Generator initializes its processors.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=work_folder) as runtime_config:
        json.dump(config, runtime_config)
        runtime_config.flush()
        generator = Generator(config=runtime_config.name)

    ray.config.enabled = False
    ray.init(address="auto")

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
        output_pcap = Path(os.environ.get("OUTPUT_PCAP", str(datasets_dir / "netshare.pcap")))

        def sample_batch(_count: int, directory: Path, index: int):
            batch_work = work_folder if index == 1 else directory
            if index > 1:
                generator._model_config["generation_seed"] = derive_seed(args.seed, "generation", index)
                # Fresh generation uses the same trained weights; isolate each
                # batch so old generated CSVs cannot be mistaken for new output.
                generated = directory / "generated_data"
                post = directory / "post_processed_data"
                generated.mkdir()
                post.mkdir()
                if not generator._generate(
                    input_train_data_folder=generator._get_pre_processed_data_folder(str(work_folder)),
                    input_model_folder=generator._get_model_folder(str(work_folder)),
                    output_syn_data_folder=str(generated),
                    log_folder=generator._get_generated_data_log_folder(str(work_folder)),
                ):
                    raise RuntimeError("Failed to sample additional NetShare output")
                if not generator._post_process(
                    input_folder=str(generated), output_folder=str(post),
                    log_folder=generator._get_post_processed_data_log_folder(str(work_folder)),
                    pre_processed_data_folder=generator._get_pre_processed_data_folder(str(work_folder)),
                ):
                    raise RuntimeError("Failed to post-process additional NetShare output")
            yield from iter_best_csv_pcaps(batch_work, directory)

        generate_to_size(reference_pcap, output_pcap, sample_batch, target_bytes,
                         batch_units=1, max_batches=args.max_generation_batches)
        validate_readable_pcap(output_pcap)

    ray.shutdown()


if __name__ == "__main__":
    main()
