from tests import _PATH_DATA, _PROJECT_ROOT
import torch
from src.data.mnist import load
from src.models.predict_model import MyAwesomeModel

print(_PROJECT_ROOT + "/models/trained_model.pt")


input_train, labels_train, test_inputs, test_labels = load(input_path=_PATH_DATA +"/processed")
a: MyAwesomeModel = torch.load(_PROJECT_ROOT + "/models/trained_model.pt")

assert a.forward(input_train[:10]).shape == input_train[:10].shape