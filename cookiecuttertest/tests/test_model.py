from tests import _PATH_DATA, _PROJECT_ROOT
import torch
from src.data.mnist import load
from src.models.predict_model import MyAwesomeModel
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists(_PROJECT_ROOT + "/models/trained_model.pt"), reason="Data files not found")
def test_output_size():
    input_train, labels_train, test_inputs, test_labels = load(input_path=_PATH_DATA +"/processed")
    a: MyAwesomeModel = MyAwesomeModel()
    a.load_state_dict(torch.load(_PROJECT_ROOT + "/models/trained_model.pt"))
    print(input_train[:10].shape)
    assert a.forward(input_train[:10].reshape(input_train[:10].shape[0], -1)).shape == torch.zeros((10, 10)).shape