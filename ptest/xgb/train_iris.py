import pandas as pd
import xgboost as xgb


# download census dataset
# gcloud storage cp --recursive gs://cloud-samples-data/ai-platform/iris .


def main():
    # Download data
    iris_data_filename = "iris/iris_data.csv"
    iris_target_filename = "iris/iris_target.csv"
    # data_dir = 'gs://cloud-samples-data/ai-platform/iris'

    # Load data into pandas, then use `.values` to get NumPy arrays
    iris_data = pd.read_csv(iris_data_filename).values
    iris_target = pd.read_csv(iris_target_filename).values

    # Convert one-column 2D array into 1D array for use with XGBoost
    iris_target = iris_target.reshape((iris_target.size,))

    # Load data into DMatrix object
    dtrain = xgb.DMatrix(iris_data, label=iris_target)

    # Train XGBoost model
    bst = xgb.train({}, dtrain, 20)

    # Export the classifier to a file
    model_filename = "model.json"
    bst.save_model(model_filename)


if __name__ == "__main__":
    main()
