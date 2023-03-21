import torch
import numpy as np


"""
hyperparameters
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
learning_rate = 1e-4
batch_size = 64    # batch size

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# initialize CNN Model
model = Model()
model = model.to(device)
optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 35

model.eval()

"""
Load CIFARdataset
"""
cifar = np.load('path/to/CIFAR.npz')
X,y,label_names = cifar['X'], cifar['y']*1.0, cifar['label_names']
print("Read in CIFAR10 dataset with %d examples, and labels:\n %s" % (X.shape[0], label_names))

"""
Without Transformations
"""
# transform = Compose([ToTensor()]) 
# train_dataset = CIFARDataset(X, y, "train", transform)
# train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

"""
With Transformations
"""
# RandomHorizontalFlip
transform = Compose([ToTensor(), 
                     RandomHorizontalFlip(-5), # RandomHorizontalFlip
                     RandomAffine(0.8),        # RandomAffine
                     ColorJitter(brightness=0.8, saturation=0.8)]) # ColorJitter()
train_dataset = CIFARDataset(X, y, "train", transform)
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

old_loss = 10
for t in tqdm(range(epochs)):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, old_loss, epochs, device)
    test_loop(test_dataloader, model, loss_fn, device)
print("Done!")
