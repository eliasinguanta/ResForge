import utils
import experiments

if __name__ == "__main__":
    args = utils.parse_args()
    experiments.grid_search(args, seed=42, steps=5)