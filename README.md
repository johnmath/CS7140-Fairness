# CS7140-Fairness

[Getting Started](#getting-started)
[Training a Classifier](#how-to-train-the-mlp-classifier)
## Getting Started

First, make sure that you have [PyTorch](https://pytorch.org/get-started/locally/), [NumPy](https://numpy.org/install/), [scikit-learn](https://scikit-learn.org/stable/install.html), and [pandas](https://pandas.pydata.org/docs/getting_started/index.html) installed using a package manager, such as [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [pip](https://pip.pypa.io/en/stable/).

## How to Train the MLP Classifier

In this example, we will be training a fully connected neural network (or MLP) on the UCI Adults dataset. This dataset has several categorical and continuous attributes that we will use to determine whether an individual's income is above 50K (1) or below 50K (0). 

First, we have to import the libraries that we will need to load the data, initialize a PyTorch model, and fit the model to the data.

```python
from models import NeuralNet
from utils import ModelUtils, DataUtils
import torch.optim as optim
import torch.nn as nn
```

We also import `torch.optim` and `torch.nn` since they contain our optimizer (Adam) and our loss function (Categorical Cross Entropy), respectively.

Now, we can use the **DataUtils** to load the pandas dataframe located at `Data/adult.data` and convert it into a PyTorch DataLoader.

```python
train, test = DataUtils.load_adults(use_torch_dataset=False)
train_loader = DataUtils.dataframe_to_dataloader(train, batch_size=512, using_ce_loss=True)
test_loader = DataUtils.dataframe_to_dataloader(test, batch_size=512, using_ce_loss=True)
loaders = {"train": train_loader, "test": test_loader}
```

Using the number of attributes in the training set dataframe (minus the label column), we can tell the **NeuralNet** class how many neurons should be in the first layer.

```python
input_dim = len(train.columns) - 1
model = NeuralNet(input_dim=input_dim, layer_sizes=[64])
```

Our model's architecture is the following:

```
NeuralNet(
  (layers): Sequential(
    (0): Linear(in_features=91, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=64, out_features=2, bias=True)
  )
)
```

This is a neural network with 1 hidden layer comprised of 64 neurons, a ReLU activation function, and dropout regularization.

Finally, all we have to do to train this model is call the `fit_model` function from our **ModelUtils** class

```python
out_model, train_loss, test_loss, test_acc  = ModelUtils.fit_model(dataloaders=loaders, 
                                                                   model=model,
                                                                   epochs=50,
                                                                   criterion=nn.CrossEntropyLoss(),
                                                                   optim_init=optim.Adam,
                                                                   optim_kwargs={"lr": .003}, 
                                                                   device="cuda")
```
