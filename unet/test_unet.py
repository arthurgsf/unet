import numpy as np
from unet import Unet
from copy import deepcopy


def test_model_train():
    # arrange
    model = Unet((128, 128, 1))
    w1 = deepcopy(model.weights)
    images = np.random.rand(2, 128, 128, 1)
    labels = np.random.randint(0, 2, (2, 128, 128, 1))

    # act
    model.compile("sgd", loss="binary_crossentropy", metrics="acc")
    H = model.fit(images, labels, verbose=0)

    # assert
    w2 = model.weights
    
    
    assert np.all([np.any(w1[i] != w2[i]) for i in range(len(w2))])