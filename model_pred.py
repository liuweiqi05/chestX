import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def predict_ts_ds(model, ds_path, num_pic):
    pic = os.listdir(ds_path)[:num_pic]

    fig, axs = plt.subplots(1, len(pic), figsize=(15, 5))
    for i, img in enumerate(pic):
        img_path = os.path.join(ds_path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (256, 256))
        image_arr = image.astype(np.float32) / 255.0
        image_arr = np.expand_dims(image_arr, axis=0)
        pred = model.predict(image_arr)

        axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[i].axis('off')

        if pred[0][0] <= 0.5:
            predict = 'Normal'
        else:
            predict = 'PNEUMONIA'
        axs[i].set_title(f'Prediction: {predict}')


plt.tight_layout()
plt.show()
