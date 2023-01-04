import matplotlib.pyplot as plt
import torch
from predict_model import MyAwesomeModel

model = MyAwesomeModel()
train_set = torch.load("data/processed/inputs")
labels = torch.load("data/processed/labels")
input, labels = torch.tensor(train_set[0]), torch.tensor(train_set[1])
all_loss = []
for epoch in range(100):
    running_loss = 0
    for i in range(len(input)//50):
        images = input[50 * i: 50 * (i + 1)]
        labs = labels[50 * i: 50 * (i + 1)]
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
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
torch.save(model, "models/trained_model.pt")