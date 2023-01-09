import matplotlib.pyplot as plt
import torch
from predict_model import MyAwesomeModel
from src.data.mnist import load

model = MyAwesomeModel()
input_train, labels_train, _, _ = load("data/processed")
all_loss = []
for epoch in range(5):
    running_loss = 0
    for i in range(len(input_train)//50):
        images = input_train[50 * i: 50 * (i + 1)]
        labs = labels_train[50 * i: 50 * (i + 1)]
        # Flatten MNIST images into a 784 long vector
        images = images.reshape(images.shape[0], -1)
        # TODO: Training pass
        output = model.forward(images)
        loss = model.criterion(output, labs.to(torch.long))
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / i}")
        all_loss.append(running_loss)
plt.plot(all_loss)
plt.show()
torch.save(model.state_dict(), "models/trained_model.pt")