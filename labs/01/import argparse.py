import argparse
import pandas as pd
import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    # Convert the dataset to a DataFrame
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['bias'] = 1
    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    df.target = dataset.target
    X_train, X_test, target_train, target_test = sklearn.model_selection.train_test_split(
    df.values,  # Input data (including the bias column)
    df.target,          # Target values
    test_size = args.test_size,   # Use test_size from the argument parser
    random_state = args.seed      # Set random_state for reproducibility
    )
    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ target_train
    # TODO: Predict target values on the test set.
    predict = X_test @ W
    # TODO: Manually compute root mean square error on the test set predictions.
    rmse = np.sqrt(np.mean((predict - target_test)**2))

    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
