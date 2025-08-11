# main.py
from src.model import Sequential
from src.layers import Dense
from src.activation import ReLU
from src.trainer import Trainer
from src.optimizers import SGD
from src.utils.data import load_digits_data
from src.utils.io import save_model
from src.logger import get_logger

def build_model(input_dim: int, hidden: int, output_dim: int):
    model = Sequential()
    model.add(Dense(input_dim, hidden))
    model.add_activation(ReLU())
    model.add(Dense(hidden, output_dim))   # final dense -> logits
    # no activation added for final layer; loss will apply softmax
    return model

if __name__ == "__main__":
    logger = get_logger("ffnn")
    X_train, X_test, y_train, y_test = load_digits_data()
    n_classes = y_train.shape[1]
    model = build_model(X_train.shape[1], 64, n_classes)
    optim = SGD(lr=0.1, momentum=0.9)  # try lr=0.1, momentum helps
    trainer = Trainer(model, optim)
    history = trainer.fit(X_train, y_train, epochs=30, batch_size=64, X_val=X_test, y_val=y_test)
    save_model("model.npz", model.params())
    logger.info("Training finished and model saved as model.npz")
