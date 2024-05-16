import matplotlib.pyplot as plt

class PlotLRS():
    def __init__(self, train_losses, val_losses, FILE_DELIN) -> None:
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.FILE_DELIN = FILE_DELIN
        
    def plot(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig(f'elbo{self.FILE_DELIN}.png', dpi=600)
