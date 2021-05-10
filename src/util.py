import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import dataSet

DIR = '../data/'

def load_data():
    df  = pd.read_csv(DIR+'train.csv', dtype=np.float32)
    y = df['label'].values 
    X = df.drop(['label'], 1).values/255
    X = X.reshape((-1, 1, 28, 28))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    return convert_to_Tensors([X_train, X_test, y_train, y_test])

def convert_to_Tensors(data):
    y_train = torch.from_numpy(data[2]).type(torch.LongTensor)
    y_test = torch.from_numpy(data[3]).type(torch.LongTensor)
    my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    train_dataset = dataSet.CustomDataset(data[0], y_train, my_transforms)
    test_dataset = dataSet.CustomDataset(data[1], y_test, my_transforms)

    return train_dataset, test_dataset

def fit(model, train_loader,test_loader, epochs, batch_size, error, optimizer):
    for epoch in range(epochs):
        correct = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()
        
        print('[util]: ','Epoch : {} \tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
            epoch, loss.data, float(correct*100) / float(len(train_loader.dataset))))
        evaluate(model, test_loader, batch_size=batch_size)


def evaluate(model, test_loader, batch_size):
    correct = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = Variable(test_imgs).float()
        with torch.no_grad():
            output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print('[util]: ',"Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*batch_size)))

def save_model(model, filename):
    torch.save(model.state_dict(), DIR+'models/'+filename+'.pth')
    print('[util]: Model saved, as ',DIR+'models/'+filename+'.pth')

def load_model(model, filename):
    model.load_state_dict(torch.load(DIR+'models/'+filename+'.pth'))
    model.eval()
    print('[util]: Model loaded, from ',DIR+'models/'+filename+'.pth')
