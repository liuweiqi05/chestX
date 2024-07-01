import dataset_proc as data_p
import model as ML
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import model_test as MLT
import model_pred as MLP
from sklearn.preprocessing import LabelEncoder

# A = data_p.train_d('D:/chest_xray/train')
# B = data_p.train_d('D:/chest_xray/test')
# C = data_p.train_d('D:/chest_xray/val')
# print(A.shape)
# print(B.shape)
# print(C.shape)
D, E = data_p.init_train_ds('D:/chest_xray/train')
F = data_p.init_test_ds('D:/chest_xray/test/')
print(D)
labels = ['NORMAL', 'PNEUMONIA']
LabelEncoder().fit(labels)
tr_model = ML.init_model()
tr_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = tr_model.fit_generator(D,
                                 epochs=1,
                                 validation_data=E,
                                 callbacks=early_stopping)

# tsds_path = 'D:/chest_xray/test/PNEUMONIA/'
# MLP.predict_ts_ds(tr_model, tsds_path, 5)

val_loss, val_acc = tr_model.evaluate(E)

print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)

best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1

plt.style.use('ggplot')

fig, axs = plt.subplots(1, 2, figsize=(16, 5))

axs[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
axs[0].scatter(best_epoch - 1, history.history['val_accuracy'][best_epoch - 1], color='yellow',
               label=f'Best Epoch: {best_epoch}')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Training and Validation Accuracy')
axs[0].legend()

axs[1].plot(history.history['loss'], label='Training Loss', color='blue')
axs[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
axs[1].scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='yellow',
               label=f'Best Epoch: {best_epoch}')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Training and Validation Loss')
axs[1].legend()

plt.tight_layout()
plt.show()

MLT.predict_ds(tr_model, F, labels, 20, 5)

