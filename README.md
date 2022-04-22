# CS7140-Fairness

[Getting Started](#getting-started)

[Training a Classifier With PyTorch](#how-to-train-an-mlp-classifier-using-pytorch)

[Training a Classifier With scikit-learn](#how-to-train-an-mlp-classifier-using-scikit)

## Getting Started

First, make sure that you have [PyTorch](https://pytorch.org/get-started/locally/), [NumPy](https://numpy.org/install/), [scikit-learn](https://scikit-learn.org/stable/install.html), and [pandas](https://pandas.pydata.org/docs/getting_started/index.html) installed using a package manager, such as [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [pip](https://pip.pypa.io/en/stable/).

## How to Train an MLP Classifier using scikit

In this example, we will be training a fully connected neural network (or MLP) on the UCI Adults dataset. This dataset has several categorical and continuous attributes that we will use to determine whether an individual's income is above 50K (1) or below 50K (0). 

First, we have to import the libraries that we will need to load the data, initialize a scikit-learn model, and fit the model to the data.

```python
from sklearn.neural_network import MLPClassifier
from utils import DataUtils
```


Now, we can use the **DataUtils** to load the pandas dataframe located at `Data/adult.data`. We also initialize our model with hyperparameters.

```python
df_train, df_test = DataUtils.load_adults(use_torch_dataset=False)

model = MLPClassifier(hidden_layer_sizes=(64), 
                      learning_rate_init=.003, 
                      verbose=True, 
                      learning_rate="adaptive")
```

This is a neural network with 1 hidden layer comprised of 64 neurons and a ReLU activation function.

Finally, all we have to do to train this model is call the `fit` function that is built into the scikit-learn model. `X` contains all the columns of the dataframe except for the labels. `y` contains the labels.

```python
trained_model = model.fit(X=df_train.loc[:, df_train.columns != "output"], 
                          y=df_train["output"])
```


### Getting Y_pred, Y_true, and A as vectors

To compute our fairness metrics, we need the model's binary predictions, the ground truth labels, and the sensitive class. To do this, we can use the `predict` function that is built into the scikit-learn model.

First, import numpy so that we can convert everything to np.arrays (vectors)

```python
import numpy as np
```

Then, call the `predict` function on the test data, convert the true labels from the test data into an array, and convert the sensitive attribute column into an array.


```python
y_pred = trained_model.predict(df_test.iloc[:, df_test.columns != "output"])
y_true = np.array(df_test["output"])
A = np.array(df_test["race_ Black"])
```

Note, the features in the one-hot version of adults have columns that are formatted as `<category>_ <attribute>`.

## How to Train an MLP Classifier using PyTorch

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
