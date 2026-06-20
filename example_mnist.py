import numpy as np
import matplotlib.pyplot as plt

def main():
    dataset = np.load('emnist_perfect_16x16.npz')
    X = dataset['X']
    y = dataset['y']
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()
    
    for i in range(10):
        axes[i].imshow(X[i], cmap='gray')
        
        char_label = chr(y[i])
        axes[i].set_title("Label: " + char_label)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()