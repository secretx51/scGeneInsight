import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay)

class Evaluation():
    def __init__(self, model, test_dataloader, device, FILE_DELIN) -> None:
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.FILE_DELIN = FILE_DELIN
        
        # Evaluate model ON INIT
        self.accuracy, self.report, self.cm = self._evaluateMetrics()
    
    def _evaluate(self):
        self.model.eval()
        true_values = []
        predicted_values = []

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.autocast(device_type='cuda'):
                    outputs = self.model(inputs.float())
                _, predicted = torch.max(outputs, 1)
                true_values.extend(labels.cpu().numpy())
                predicted_values.extend(predicted.cpu().numpy())

        return true_values, predicted_values

    def _calcMetrics(self, true_values, predicted_values):
        accuracy = accuracy_score(true_values, predicted_values)
        report = classification_report(true_values, predicted_values)
        cm = confusion_matrix(true_values, predicted_values)
        return accuracy, report, cm

    def _evaluateMetrics(self):
        true_values, predicted_values = self._evaluate()
        return self._calcMetrics(true_values, predicted_values)
    
    def getAccuracy(self):
        return self.accuracy
        
    def displayMetrics(self):
        print(f'Accuracy: {self.accuracy * 100:.2f}%')
        print(self.report)
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        disp.plot()
        plt.savefig(f'confusion_matrix{self.FILE_DELIN}.png')
    
    def saveModel(self):
        torch.save(self.model.state_dict(), f'linear_b2m_regressor{self.FILE_DELIN}.pth')
