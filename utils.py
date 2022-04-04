import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def dataframe_to_torch_dataset(dataframe):
    """Convert a pandas dataframe to a PyTorch Dataset of Tensor objects"""

    new = dataframe.copy()
    # Take last column as prediction targets
    label_column = new.columns[-1]
    labels = torch.Tensor(pd.DataFrame(new[label_column]).values)
    del new[label_column]
    data = torch.Tensor(new.values)

    return torch.utils.data.TensorDataset(data, labels)

def datasets_to_dataloaders(train, test, batch_size=256, num_workers=4, shuffle=True):
    """Wrap a PyTorch dataset in a Dataloader (to allow batch computations)"""
    train_loader = torch.utils.data.DataLoader(train, 
                                               batch_size=batch_size, 
                                               num_workers=num_workers, 
                                               shuffle=shuffle)
    
    test_loader = torch.utils.data.DataLoader(test, 
                                              batch_size=batch_size, 
                                              num_workers=num_workers, 
                                              shuffle=shuffle)
    
    return {"train": train_loader, "test": test_loader}
    

def load_adults(root="Data/", test=0.25, use_torch_dataset=True):
    """Cleans and loads UCI Adults dataset"""
    
    column_names = ["age",
                "workclass", 
                "fnlwgt", 
                "education", 
                "education-num", 
                "marital-status", 
                "occupation", 
                "relationship", 
                "race", 
                "sex", 
                "capital-gain", 
                "capital-loss", 
                "hours-per-week", 
                "native-country"]
    
    uncleaned_df = pd.read_csv("Data/adult.data", header=None)
    
    # Map the column names to each column in the dataset
    mapping = {i:column_names[i] for i in range(len(column_names))}
    mapping[14] = "output"
    uncleaned_df = uncleaned_df.rename(columns=mapping)
    
    # Columns with categorical data
    cat_columns = ["workclass", 
                "education", 
                "marital-status", 
                "occupation", 
                "relationship", 
                "race", 
                "sex",  
                "native-country"]

    cont_columns = ["age",
                    "fnlwgt",
                    "education-num",
                    "capital-gain",
                    "capital-loss",
                    "hours-per-week"]

    dummy_tables = [pd.get_dummies(uncleaned_df[column], prefix=column) for column in cat_columns]
    dummy_tables.append(uncleaned_df.drop(labels=cat_columns, axis=1))

    df = pd.concat(dummy_tables, axis=1)
    encoder = LabelEncoder()
    df["output"] = encoder.fit_transform(df["output"])
    df[cont_columns] = df[cont_columns]/df[cont_columns].max()
    train, test = train_test_split(df, test_size=test, shuffle=True)
    
    if use_torch_dataset:
        return dataframe_to_torch_dataset(train), dataframe_to_torch_dataset(test)
    else:
        return train, test

    
def fit_model(dataloaders, model, epochs=100, optim_init=optim.SGD, lr=.01, criterion=nn.BCEWithLogitsLoss(), device="cpu", verbose=True):
    """Fits a PyTorch model to any given dataset"""
    
    model = model.to(device)
    optimizer = optim_init(model.parameters(), lr=lr)
    train_error = []
    test_acc = []
    print("Training...")
    if verbose:
        print("-"*8)
    

    for epoch in range(1, epochs + 1):
        if verbose:
            print(f"Epoch {epoch}")
            
        running_train_loss = 0
        running_test_loss= 0 
        
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                for (inputs, labels) in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item() * inputs.size(0)

            elif phase == "test":
                model.eval()
                for (inputs, labels) in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item() * inputs.size(0)
                    
        train_error.append(running_train_loss/len(dataloaders["train"].dataset))
        test_acc.append(running_test_loss/len(dataloaders["test"].dataset))
        if verbose:
            print(f"Train Error: {train_error[-1]:.4}")
            print(f"Test Error: {test_acc[-1]:.4}")
            print("-"*8)
    model.eval()
    return model, train_error, test_acc