import pickle

def load_encoder(encoder_path):
    with open(encoder_path, "rb") as encoder_file: 
        encoder = pickle.load(encoder_file)
        
    return encoder


def compute_graphics(train_log, save_path):
    import pandas as pd
    import os
    from matplotlib import pyplot as plt    

    df = pd.read_csv(train_log)

    # Extract data from the DataFrame
    epochs = df['epoch']
    accuracy = df['accuracy']
    loss = df['loss']
    val_accuracy = df['val_accuracy']
    val_loss = df['val_loss']

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    # plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    # plt.show()