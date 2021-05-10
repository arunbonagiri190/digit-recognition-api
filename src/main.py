import torch
import torch.nn as nn
import torch.utils.data as tud
import torch.optim as optim
import util
import model

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

train_dataset, test_dataset = util.load_data()

train_loader = tud.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = tud.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = model.CNN()
print('[main]: ',model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
error = nn.CrossEntropyLoss()

util.fit(model, train_loader=train_loader,test_loader=test_loader, epochs=EPOCHS, batch_size=BATCH_SIZE, error=error, optimizer=optimizer)
#util.save_model(model, 'model_b_32_e_10')