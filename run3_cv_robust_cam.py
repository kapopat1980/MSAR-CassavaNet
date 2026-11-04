import numpy as np
import cv2
import torch
import torch.nn as nn

class GradCAM:
    """Grad-CAM visual attribution module for spatial interpretability."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        output = self.model(input_image)
        self.model.zero_grad()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-8)
        return cam

if __name__ == "__main__":
    print("Grad-CAM visualization utility initialized.")
