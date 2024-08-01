import torch.nn as nn
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import pytorch_lightning as pl
import torch

class ImageClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
                
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        self.cross_entropy = cross_entropy
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    
    # will be used during inference
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
        gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
        acc = self.accuracy(output, gts)
        
        self.log('test_acc', acc)
        
        self.test_gts = gts
        self.test_output = output
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
