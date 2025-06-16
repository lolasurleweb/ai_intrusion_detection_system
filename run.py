import argparse
from src.data.preprocessing import preprocess
from src.training.train_tabnet import run_training
from src.utils.seeding import set_seed
from simulate_deployment import run_deployment_simulation_ensemble
from src.training.evaluate_tabnet import run_final_test_model_ensemble

def main():
    parser = argparse.ArgumentParser(description="Cybersecurity ML-Pipeline")
    parser.add_argument("step", choices=["preprocess", "train", "test", "simulate_deployment"],
                        help="Wähle den Teil der Pipeline, den du ausführen willst.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entscheidungsschwelle für attack_detected bei simulate_deployment (default=0.5)")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.15,
                    help="Unsicherheits-Schwelle für Ensemble-Entscheidungen (default=0.15)")
    args = parser.parse_args()

    if args.step == "preprocess":
        preprocess()
    elif args.step == "train":
        set_seed(42)
        run_training()
    elif args.step == "test":
        run_final_test_model_ensemble()
    elif args.step == "simulate_deployment":
        run_deployment_simulation_ensemble(threshold=args.threshold, uncertainty_threshold=args.uncertainty_threshold)

if __name__ == "__main__":
    main()
