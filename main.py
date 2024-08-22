import torch
from torchmetrics import Accuracy,F1Score,ConfusionMatrix,Recall,Precision
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layers = nn.Sequential(  #  DEFINE MODEL LAYERS
            nn.Linear(5,5),
            nn.GELU(),
            nn.Linear(5,4),
            nn.GELU(),
            nn.Linear(4,4),
            nn.GELU(),    
            nn.Linear(4,2),
            nn.Sigmoid()
        )
    def forward(self,x:torch.Tensor)->torch.Tensor: #DEFINE FORWARD COMPUTATION
        return self.layers(x) 

#CREATE DATASET
class Trainset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self,index):
        pass
    def __len__(self):
        pass


trainset  = Trainset()
testset = Trainset()
# CREATE DATALOADER
trainloader = DataLoader(trainset, batch_size=4,shuffle=True)

testloader = DataLoader(testset,batch_size=4,shuffle=False)

model = NN()
# DEFINE LOSS AND OPTIMIZER
criterion = nn.BCELoss()

opimizer = optim.Adam(model.parameters(), lr=1e-5)
accuracy = Accuracy(task='binary')
# Precision metric for binary classification
precision = Precision(task="binary")
f1 = F1Score(task="binary")
recall = Recall(task="binary")
confusion_matrix = ConfusionMatrix(task="binary")


#DEFINE TRAIN LOOP
epochs = 2

for epoch in  range(1,epochs+1):
    # train 
    for batch_idx,features,labels in enumerate(trainloader):
        opimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits,labels)
        loss.backward()
        opimizer.step()
        if batch_idx % 100 == 0:
            print("batch {} : loss = {}".format(epoch*batch_idx,loss.detach().item()))
            
    for batch_idx,features,labels in enumerate(testloader):  
        logits = model(features)
        
        accuracy.update(logits, labels)
        precision.update(logits, labels)
        recall.update(logits, labels)
        f1.update(logits, labels)
        confusion_matrix.update((logits >= 0.5).long(), labels)
        
        epoch_accuracy = accuracy.compute()
        epoch_precision = precision.compute()
        epoch_recall = recall.compute()
        epoch_f1 = f1.compute()
        epoch_confusion_matrix = confusion_matrix.compute()
        

        accuracy.reset()
        precision.reset()
        recall.reset()
        f1.reset()
        confusion_matrix.reset()

        print(f'Accuracy: {epoch_accuracy.item()}')
        print(f'Precision: {epoch_precision.item()}')
        print(f'Recall: {epoch_recall.item()}')
        print(f'F1 Score: {epoch_f1.item()}')
        print(f'Confusion Matrix:\n{epoch_confusion_matrix}')


#export model


model.eval()

dummy_input = torch.randn(4,5)

torch.onnx.export(
    model=model,
    args=dummy_input,
    f='model.onnx',
    input_names=['input'],
    output_names=["output"],
    export_params=True,
    verbose=True
)


