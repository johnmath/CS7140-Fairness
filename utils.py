import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



class DataUtils:
    
    def __init__(self):
        pass
    
    @staticmethod
    def dataframe_to_torch_dataset(dataframe, using_ce_loss=False):
        """Convert a one-hot pandas dataframe to a PyTorch Dataset of Tensor objects"""

        new = dataframe.copy()
        label = list(new.columns)[-1]
        labels = torch.Tensor(pd.DataFrame(new[label]).values)
        del new[label]
        data = torch.Tensor(new.values)

        if using_ce_loss:
            # Fixes tensor dimension and float -> int if using cross entropy loss
            return torch.utils.data.TensorDataset(data, labels.squeeze().type(torch.LongTensor))
        else:
            return torch.utils.data.TensorDataset(data, labels)
        
    @staticmethod
    def dataset_to_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True):
        """Wrap PyTorch dataset in a Dataloader (to allow batch computations)"""
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, 
                                             num_workers=num_workers, 
                                             shuffle=shuffle)
        return loader

    @staticmethod
    def dataframe_to_dataloader(dataframe, batch_size=256, num_workers=4, shuffle=True, using_ce_loss=False):
        """Convert a pandas dataframe to a PyTorch Dataloader"""

        dataset = DataUtils.dataframe_to_torch_dataset(dataframe, using_ce_loss=using_ce_loss)
        return DataUtils.dataset_to_dataloader(dataset, 
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     shuffle=shuffle)

    @staticmethod
    def load_adults(root="Data/", test=0.25, use_torch_dataset=True):
        """Cleans and loads UCI Adults dataset"""

        column_names = [
            "age",
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
            "native-country"
        ]

        uncleaned_df = pd.read_csv("Data/adult.data", header=None)

        # Map the column names to each column in the dataset
        mapping = {i:column_names[i] for i in range(len(column_names))}
        mapping[14] = "output"
        uncleaned_df = uncleaned_df.rename(columns=mapping)

        # Columns with categorical data
        cat_columns = [
            "workclass", 
            "education", 
            "marital-status", 
            "occupation", 
            "relationship", 
            "race", 
            "sex",  
            "native-country"
        ]

        cont_columns = [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week"
        ]

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

class ModelUtils:
    
    def __init__(self):
        pass
    
    @staticmethod
    def fit_model(dataloaders, 
                  model, epochs=100, 
                  optim_init=optim.SGD, 
                  optim_kwargs={"lr": 0.003, "momentum": 0.9}, 
                  criterion=nn.BCEWithLogitsLoss(), 
                  device="cpu", 
                  verbose=True):
        """Fits a PyTorch model to any given dataset

            ...
            Parameters
            ----------
                dataloaders : dict
                   Dictionary containing 2 PyTorch DataLoaders with keys "train" and 
                   "test" corresponding to the two dataloaders as values
                model : torch.nn.Module (PyTorch Neural Network Model)
                    The desired model to fit to the data
                epochs : int
                    Training epochs for shadow models
                optim_init : torch.optim init object
                    The init function (as an object) for a PyTorch optimizer. 
                    Note: Pass in the function without ()
                optim_kwargs : dict
                    Dictionary of keyword arguments for the optimizer
                    init function
                criterion : torch.nn Loss Function
                    Loss function used to train model
                device : str
                    The processing device for training computations. 
                    Ex: cuda, cpu, cuda:1
                verbose : bool
                    If True, prints running loss and accuracy values

            ...
            Returns
            -------
            model : torch.nn.Module (PyTorch Neural Network Model)
                The trained model
            train_error : list
                List of average training errors at each training epoch
            test_acc : list 
                List of average test accuracy at each training epoch
        """

        model = model.to(device)
        optimizer = optim_init(model.parameters(), **optim_kwargs)
        train_error = []
        test_loss = []
        test_acc = []
        print("Training...")
        if verbose:
            print("-"*8)

        try:

            for epoch in range(1, epochs + 1):
                if verbose:
                    print(f"Epoch {epoch}")

                running_train_loss = 0
                running_test_loss = 0 
                running_test_acc = 0

                for phase in ["train", "test"]:
                    if phase == "train":
                        model.train()
                        for (inputs, labels) in dataloaders[phase]:
                            model.zero_grad()
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
                            running_test_acc += sum(torch.max(outputs, dim=1)[1] == labels) 

                train_error.append(running_train_loss/len(dataloaders["train"].dataset))
                test_loss.append(running_test_loss/len(dataloaders["test"].dataset))
                test_acc.append(running_test_acc/len(dataloaders["test"].dataset))
                if verbose:
                    print(f"Train Error: {train_error[-1]:.4}")
                    print(f"Test Error: {test_loss[-1]:.4}")
                    print(f"Test Accuracy: {test_acc[-1]*100:.4}%")
                    print("-"*8)
        except KeyboardInterrupt:
            return model, train_error, test_loss

        return model, train_error, test_loss, test_acc