import tensorflow as tf
import optuna
from censnet_linear_class import censnet_optuna
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run Optuna with CensNet optimization")
    parser.add_argument(
        "--savepath", type=str, default="./results/", help="Path to save results"
    )
    parser.add_argument(
        "--db_path", type=str, default="./db/", help="Path to the database"
    )
    parser.add_argument(
        "--method_folder",
        type=str,
        nargs="+",
        default=["/Dummy/npy/", "/Col/npy/"],
        help="Method folders",
    )
    parser.add_argument("--method", type=int, default=0, help="Method index")
    parser.add_argument(
        "--n_trials", type=int, default=100, help="Number of trials for Optuna"
    )
    parser.add_argument(
        "--study_name", type=str, default="linear", help="Study name for Optuna"
    )
    return parser.parse_args()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    args = parse_args()
    storage_name = f"sqlite:///{args.study_name}.db"
    optimizer = censnet_optuna(
        args.savepath, args.db_path, args.method_folder, args.method
    )

    if os.path.exists(f"{args.db_name}.db"):
        print(f"Resuming study from {args.db_name}.db")
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=storage_name,
            load_if_exists=True,
        )

    else:
        print(f"Starting new study, saving to {args.db_name}.db")
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=storage_name,
            load_if_exists=False,
        )

    study.optimize(
        optimizer.objective,
        n_trials=args.n_trials,
    )
