import numpy as np
import matplotlib.pyplot as plt


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def predict_ds(model, ds, class_labels, num, col):

    ds_shuffled = ds.shuffle(buffer_size=len(ds))
    pred = model.predict(ds_shuffled)

    plt.figure(figsize=(15, 10))
    for i, (images, labels) in enumerate(ds_shuffled.take(num)):
        images = images.numpy()

        for j in range(len(images)):
            if i * col + j < num:
                Prediction = class_labels[np.argmax(pred[i * col + j])]
                Truth = class_labels[np.argmax(labels[j])]

                plt.subplot(num // col + 1, col, i * col + j + 1)
                plt.imshow(images[j].astype("uint8"))
                plt.title(f'Truth: {Truth}\nPrediction: {Prediction}')
                plt.axis('off')

    plt.tight_layout()
    plt.show()
