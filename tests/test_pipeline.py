import os
import pickle

def test_model_exists():
    assert os.path.exists("models/best_model.pkl"), "Model file not found!"

def test_model_loads():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None, "Model failed to load!"
