from __future__ import annotations

import argparse
import json

from .config import load_config
from .core import read_jsonl, write_jsonl
from .nuimages import convert_nuimages
from .pipeline import (
    auto_select_masks,
    auto_verify,
    bootstrap_student,
    build_joint_training_dataset,
    build_self_training_dataset,
    evaluate_student,
    export_pseudo,
    promote_student,
    propose,
    segment,
    train_student,
    verify,
)
from .video import propagate_video


def _common_stage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="longtail-prelabel")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert")
    convert.add_argument("--config", required=True)

    for name in ("propose", "verify", "segment", "auto-verify", "auto-select"):
        stage = commands.add_parser(name)
        _common_stage(stage)
        if name == "verify":
            stage.add_argument("--resume", action="store_true")

    merge = commands.add_parser("merge-jsonl")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)

    export = commands.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--input", required=True)
    export.add_argument("--output-dir", required=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)

    evaluate = commands.add_parser("evaluate-student")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output")

    build_self_train = commands.add_parser("build-self-train")
    build_self_train.add_argument("--config", required=True)
    build_self_train.add_argument("--pseudo-dir", required=True)
    build_self_train.add_argument("--output-dir", required=True)

    build_joint_train = commands.add_parser("build-joint-train")
    build_joint_train.add_argument("--config", required=True)
    build_joint_train.add_argument("--output-dir", required=True)

    bootstrap = commands.add_parser("bootstrap-student")
    bootstrap.add_argument("--config", required=True)

    promote = commands.add_parser("promote-student")
    promote.add_argument("--config", required=True)

    propagate = commands.add_parser("propagate")
    propagate.add_argument("--config", required=True)
    propagate.add_argument("--index", required=True)
    propagate.add_argument("--seeds", required=True)
    propagate.add_argument("--output", required=True)
    propagate.add_argument("--device", default="cuda:0")
    propagate.add_argument("--shard-index", type=int, default=0)
    propagate.add_argument("--num-shards", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "convert":
        print(json.dumps(convert_nuimages(load_config(args.config)), ensure_ascii=False, indent=2))
    elif args.command in {"propose", "verify", "segment", "auto-verify", "auto-select"}:
        function = {
            "propose": propose,
            "verify": verify,
            "segment": segment,
            "auto-verify": auto_verify,
            "auto-select": auto_select_masks,
        }[args.command]
        arguments = [
            load_config(args.config),
            args.input,
            args.output,
        ]
        if args.command in {"propose", "verify", "segment"}:
            arguments.append(args.device)
        arguments.extend([args.shard_index, args.num_shards])
        if args.command == "verify":
            arguments.append(args.resume)
        function(*arguments)
    elif args.command == "merge-jsonl":
        write_jsonl(args.output, (row for path in args.inputs for row in read_jsonl(path)))
    elif args.command == "export":
        result = export_pseudo(load_config(args.config), args.input, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "train":
        print(json.dumps(train_student(load_config(args.config), args.data), indent=2))
    elif args.command == "evaluate-student":
        result = evaluate_student(load_config(args.config), args.checkpoint)
        if args.output:
            from pathlib import Path

            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "build-self-train":
        result = build_self_training_dataset(
            load_config(args.config), args.pseudo_dir, args.output_dir
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "build-joint-train":
        result = build_joint_training_dataset(load_config(args.config), args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "bootstrap-student":
        print(json.dumps(bootstrap_student(load_config(args.config)), ensure_ascii=False, indent=2))
    elif args.command == "promote-student":
        print(json.dumps(promote_student(load_config(args.config)), ensure_ascii=False, indent=2))
    elif args.command == "propagate":
        propagate_video(
            load_config(args.config),
            args.index,
            args.seeds,
            args.output,
            args.device,
            args.shard_index,
            args.num_shards,
        )


if __name__ == "__main__":
    main()
