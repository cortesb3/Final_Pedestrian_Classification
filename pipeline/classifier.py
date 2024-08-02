import torch.nn as nn
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

# PyTorch Lightning Model
class ImageClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate, num_classes):
        super().__init__()
            
        self.learning_rate = learning_rate
        self.num_classes = num_classes
         
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        self.cross_entropy = cross_entropy
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.accuracy_binary = Accuracy(task="binary")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        image, gt = batch
        output = self.forward(image)
        
        loss = self.cross_entropy(output, gt)
        acc = self.accuracy(output, gt)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return {"loss":loss, "acc":acc}
    
    def validation_step(self, batch, batch_idx):
        image, gt = batch
        output = self.forward(image)
        
        loss = self.cross_entropy(output, gt)
        acc = self.accuracy(output, gt)
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return {"val_loss":loss, "val_acc":acc}
    
    def test_step(self, batch, batch_idx):
        image, gt = batch
        output = self.forward(image)
        
        loss = self.cross_entropy(output, gt)
        return {"test_loss":loss, "outputs" : output, "gt" : gt}
        
    def test_epoch_end(self, outputs):
        output = torch.cat([x['outputs'] for x in outputs], dim=0)

        output = np.argmax(output.detach().cpu().numpy(), axis=1)
        output = torch.from_numpy(output)#.to(device='cuda')
        
        gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
        output_list = output.cpu().numpy()
        gts_list = gts.cpu().numpy()
        
        # crosschecking accuracy metric
        sklearn_acc = accuracy_score(gts_list, output_list)
        
        print(f'\n{sklearn_acc}')
        print(f'\n\n{len(output_list)} {len(gts_list)}')
        acc = self.accuracy2(output, gts)
        self.log('test_acc', acc)
        
        self.test_gts = gts
        self.test_output = output
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        # reduce lr on plateau by half 
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10, cooldown=3)
        
        lr_scheduler_config = {"scheduler" : scheduler, "interval" : "epoch", "frequency" : 3, "monitor" : "val_loss"}
        return {"optimizer" : optimizer, "lr_scheduler" : lr_scheduler_config}
       
    def lr_scheduler_step(self, scheduler, metric, *args, **kwargs):
        scheduler.step(metric)
