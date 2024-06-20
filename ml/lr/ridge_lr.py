
import logging
from typing import Optional

from numpy import ndarray
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from base import BaseLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


class RidgeLR(BaseLR):
    def __init__(self, path_to_model: Optional[str], alpha: float = 1.0):
        if path_to_model is None:
            path_to_model = "ridge_lr.pkl"
        self.path_to_model = path_to_model
        self.alpha = alpha

    def train(self, x_train: ndarray, y_train: ndarray, x_dev: Optional[ndarray] = None, y_dev: Optional[ndarray] = None):
        if x_dev is None or y_dev is None:
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        ridge_regr = Ridge(alpha=self.alpha)

        ridge_regr.fit(x_train, y_train)

        y_train_pred = ridge_regr.predict(x_train)

        logger.info("Train Evaluation")
        self.evaluate(y_train, y_train_pred)

        y_dev_pred = ridge_regr.predict(x_dev)
        logger.info("Validation Evaluation")
        self.evaluate(y_dev, y_dev_pred)

        self.save(ridge_regr, self.path_to_model)


if __name__ == "__main__":
    from dataset import diabetes_X, diabetes_y
    print("Running ridge regression...")
    _path_to_model = "../models/ridge_lr.pkl"
    ridge_lr = RidgeLR(path_to_model=_path_to_model, alpha=0.1)

    ridge_lr.train(x_train=diabetes_X, y_train=diabetes_y)
