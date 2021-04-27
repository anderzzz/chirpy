'''Bla bla

'''
import torch
from torch import nn

class MajorityVoter(nn.Module):
    '''Bla bla

    '''
    def __init__(self, ensemble_size, voting='soft'):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.voting = voting
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, class_predictions, class_truths):
        '''Bla bla

        '''
        print (class_predictions)
        print (class_truths)

        n_batches = class_predictions.shape[0] // self.ensemble_size

        all_predictions = []
        for batch_pred in torch.chunk(class_predictions, n_batches, dim=0):
            voted = batch_pred.sum(dim=0)
            voted = voted.div(self.ensemble_size)
            all_predictions.append(voted)

        preds = torch.stack(all_predictions)

        return self.cross_entropy(preds, class_truths)