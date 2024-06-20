import logging
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Optional

from numpy import ndarray
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


class BaseLR(ABC):

    @staticmethod
    def save(model, path: str):
        """Save model to disk"""
        with open(path, "wb") as file:
            pickle.dump(model, file)

        logger.info(f"Saved model to {path}")

    @staticmethod
    def load(path_to_model: str):
        """Load model from disk"""
        with open(path_to_model, "rb") as file:
            model = pickle.load(file)
        return model

    @abstractmethod
    def train(
        self, x_train: ndarray, y_train: ndarray, x_dev: Optional[ndarray] = None, y_dev: Optional[ndarray] = None
    ):
        raise NotImplementedError("Must implement train method")

    @staticmethod
    def evaluate(y_true: ndarray, y_pred: ndarray) -> Dict[str, float]:
        mse = round(mean_squared_error(y_true, y_pred), 2)
        mape = round(100 * mean_absolute_percentage_error(y_true, y_pred), 2)

        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"Mean Absolute Percentage Error: {mape}%")

        return {"mse": mse, "mape": mape}

    @staticmethod
    def predict(model, x: ndarray):
        y_predict = model.predict(x)

        return y_predict

    def __call__(self, model, x: ndarray, *args, **kwargs):
        print("invoke __call_ method...")
        return self.predict(model, x)
