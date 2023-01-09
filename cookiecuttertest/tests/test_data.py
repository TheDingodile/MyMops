from tests import _PATH_DATA
from src.data.mnist import load


input_train, labels_train, test_inputs, test_labels = load(input_path="data/processed")


def test_data_size():
    assert len(input_train) == 25000
    assert len(labels_train) == 25000
    assert len(test_inputs) == 5000
    assert len(test_labels) == 5000
# assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
# assert that all labels are represented