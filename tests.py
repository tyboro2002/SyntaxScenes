import matplotlib.pyplot as plt
import tensorflow as tf


def show_training_images():
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # Display the first 10 images in the dataset
    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Call the function to display the images
show_training_images()