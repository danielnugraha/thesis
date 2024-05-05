from mvs import MVS
import xgboost as xgb

class MVSAdaptive(MVS):

    def __init__(self, objective, lambda_rate = 0.1, sample_rate = 0.1) -> None:
        self.lambda_rate = lambda_rate
        self.sample_rate = sample_rate
        self.objective = objective


def calculate_mean_leaf_value(bst: xgb.Booster):
    leaf_values = []
    num_leaves = 0
    sum_leaves = 0

    trees_dump = bst.get_dump(dump_format="text")
    print(trees_dump[-1])
    for line in trees_dump[-1].split("\n"):
        if "leaf" in line:
            parts = line.split(",")
            for part in parts:
                if "leaf=" in part:
                    leaf_value = float(part.split('=')[1])
                    leaf_values.append(leaf_value)
                    num_leaves += 1
        

    return leaf_values