import logging
from typing import Optional

from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ml.lr.base import BaseLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


class SimpleLR(BaseLR):
    def __init__(self, path_to_model: Optional[str]):
        if path_to_model is None:
            path_to_model = "simple_lr.pkl"
        self.path_to_model = path_to_model

    def train(
            self, x_train: ndarray, y_train: ndarray, x_dev: Optional[ndarray] = None, y_dev: Optional[ndarray] = None
    ):
        if x_dev is None or y_dev is None:
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        regr = LinearRegression()

        regr.fit(x_train, y_train)

        y_train_pred = regr.predict(x_train)

        logger.info("Train Evaluation")
        self.evaluate(y_train, y_train_pred)

        y_dev_pred = regr.predict(x_dev)
        logger.info("Validation Evaluation")
        self.evaluate(y_dev, y_dev_pred)

        # save model to disk
        self.save(regr, self.path_to_model)


if __name__ == "__main__":
    from ml.lr.dataset import diabetes_X

    _path_to_model = "./models/simple_lr.pkl"
    # init linear model
    lr = SimpleLR(path_to_model=_path_to_model)

    # training model
    # lr.train(x_train=diabetes_X, y_train=diabetes_y)

    # Load model for test
    model = lr.load(_path_to_model)

    x_test = diabetes_X[:1, :]

    y_predict = lr(x=x_test, model=model)

    print("y_predict: ", y_predict)
