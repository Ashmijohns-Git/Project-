#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train_path="C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/train"
x_train=[]
train_labels = []
classes = 7
for folder in os.listdir(train_path):

    sub_path=train_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))
        x_train.append(img_arr)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(img_arr)

train_x=np.array(x_train)
train_labels = np.array(train_labels)
print(train_x.shape, train_labels.shape) 


# In[3]:


test_path="C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/test"
x_test=[]
test_labels = []
classes=7
for folder in os.listdir(test_path):

    sub_path=test_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))

        x_test.append(img_arr)
        test_x=np.array(x_test)
        test_labels = np.array(test_labels)
print(test_x.shape, test_labels.shape)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(img_arr)
train_x=train_x/255.0
test_x=test_x/255.0


# In[4]:


training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
training_set = training_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'sparse')

train_y=training_set.classes
test_y=test_set.classes
training_set.class_indices
train_y.shape,test_y.shape


# In[5]:


IMAGE_SIZE = (224, 224)

# Define VGG19 model
vgg = VGG19(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
#adding output layer.Softmax classifier is used as it is multi-class classification
prediction = Dense(7, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
print(model.summary())
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model
# fit the model


history = model.fit(
  train_x,
  train_y,
  epochs=2,
  callbacks=[early_stop],
  batch_size=32,shuffle=True,validation_data=test_set,validation_steps=len(test_set))


# In[6]:


score = model.evaluate(test_x, test_y)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[7]:


model.save("C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/vgg")
model.summary()


# In[8]:


# accuracies

plt.plot(history.history['accuracy'], label='train acc')

plt.plot(history.history['val_accuracy'], label='val acc')

plt.legend()

plt.savefig('vgg-acc-rps-1.png')

plt.show()


# In[9]:


# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('vgg-loss-rps-1.png')
plt.show()


# In[10]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import seaborn as sns
y_pred_probabilities = model.predict(test_x)
y_pred = np.argmax(y_pred_probabilities, axis=1)  # Convert probabilities to class labels
test_labels  = ['cardboard', 'compost','glass','metal','paper','plastic','trash']
# Create confusion matrix
conf_matrix = confusion_matrix(test_y, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(classification_report(test_y, y_pred, target_names=test_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',xticklabels=test_labels, yticklabels=test_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Vgg')
plt.show()


# In[11]:


# to identify the accuracy of each class 
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)


# Overall accuracy
print ("Cardboard    compost   Glass     Metal     Paper     Plastic    Trash")
ACC = (TP+TN)/(TP+FP+FN+TN)
print (ACC)
Precision=(TP)/(TP+FP) 
print (Precision)
Recall= (TP)/(TP+FN)
print (Recall)
f1_score=(2 * Precision + Recall) / (Precision * Recall)
print (f1_score)


# In[12]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming you have calculated the metrics for each class and stored them in lists or arrays

# List of classes (replace with your class names)
classes = ["Cardboard", "compost", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Example metrics for illustration (replace these with your actual metrics)
accuracy = np.array([0.90425532, 0.9822695,  0.87411348, 0.90780142, 0.82801418, 0.87056738, 0.94503546])
precision = np.array([0.975,      0.85,       0.63768116 ,0.62626263, 0.54901961, 0.69318182, 0.75])
recall = np.array([0.42391304, 0.89473684, 0.48888889, 0.80519481, 0.95726496, 0.57009346, 0.41860465])
f1_score = np.array([0.574358974, 0.341176471, 0.565909091, 0.408064516, 0.391071429, 0.495081967, 0.611111111])

# Plotting the metrics for each class
x = np.arange(len(classes))
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 7))

# Plotting accuracy
rects1 = ax.bar(x - width, accuracy, width, label='Accuracy')
# Plotting precision
rects2 = ax.bar(x, precision, width, label='Precision')
# Plotting recall
rects3 = ax.bar(x + width, recall, width, label='Recall')
# Plotting F1-score
rects4 = ax.bar(x + 2 * width, f1_score, width, label='F1-score')

# Adding labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Metrics by Class for VGG19')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Show plot
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model



# Load your trained model (assuming you've already trained and saved it)
model = load_model('C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/vgg')  # Replace 'your_model_path_here' with the actual path

# Assuming you have the test data ('test_images') and true labels ('test_labels')
# Replace these with your actual test data and labels

# Make predictions on the test data

# Mapping class indices to their respective labels
preds = model.predict(test_x)

# Mapping class indices to their respective labels
class_labels = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Display images with predicted and true labels
plt.figure(figsize=(15, 10))
for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.title('Pred: {}'.format(class_labels[np.argmax(preds[i])], class_labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[14]:


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input
IMAGE_SIZE = (224, 224)

# Define InceptionResNetV2 model
InceptionResNetV2 = InceptionResNetV2(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)
for layer in InceptionResNetV2.layers:
    layer.trainable = False
x = Flatten()(InceptionResNetV2.output)
#adding output layer.Softmax classifier is used as it is multi-class classification
prediction = Dense(7, activation='softmax')(x)

model = Model(inputs=InceptionResNetV2.input, outputs=prediction)

# view the structure of the model
print(model.summary())
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)


# In[15]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model
# fit the model
history = model.fit(
  train_x,
  train_y,
  epochs=2,
  callbacks=[early_stop],
  batch_size=32,shuffle=True,validation_data=test_set,validation_steps=len(test_set))


# In[16]:


# accuracies

plt.plot(history.history['accuracy'], label='train acc')

plt.plot(history.history['val_accuracy'], label='val acc')

plt.legend()

plt.savefig('InceptionResNetV2-acc-rps-1.png')

plt.show()


# In[17]:


# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('InceptionResNetV2-loss-rps-1.png')
plt.show()


# In[18]:


score = model.evaluate(test_x, test_y)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[19]:


model.save("C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/InceptionResNetV2")
model.summary()


# In[20]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import seaborn as sns
y_pred_probabilities = model.predict(test_x)
y_pred = np.argmax(y_pred_probabilities, axis=1)  # Convert probabilities to class labels
test_labels  = ['cardboard', 'compost','glass','metal','paper','plastic','trash']
# Create confusion matrix
conf_matrix = confusion_matrix(test_y, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(classification_report(test_y, y_pred, target_names=test_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',xticklabels=test_labels, yticklabels=test_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of InceptionResNetV2')
plt.show()


# In[21]:


# to identify the accuracy of each class 
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)


# Overall accuracy
print ("Cardboard    compost   Glass     Metal     Paper     Plastic    Trash")
ACC = (TP+TN)/(TP+FP+FN+TN)
print (ACC)
Precision=(TP)/(TP+FP) 
print (Precision)
Recall= (TP)/(TP+FN)
print (Recall)
f1_score=(2 * Precision + Recall) / (Precision * Recall)
print (f1_score)


# In[22]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming you have calculated the metrics for each class and stored them in lists or arrays

# List of classes (replace with your class names)
classes = ["Cardboard", "compost", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Example metrics for illustration (replace these with your actual metrics)
accuracy = np.array([0.93794326, 0.96808511,  0.90425532, 0.89716312, 0.93085106, 0.88297872, 0.93971631])
precision = np.array([0.85185185,      0.67857143,       0.95 ,0.57480315, 0.90625, 0.67826087, 0.59183673])
recall = np.array([0.75, 1., 0.42222222, 0.94805195, 0.74358974, 0.72897196, 0.6744186])
f1_score = np.array([0.384057971, 0.347368421, 0.578947368, 0.384931507, 0.379310345, 0.421794872, 0.465517241])

# Plotting the metrics for each class
x = np.arange(len(classes))
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 7))

# Plotting accuracy
rects1 = ax.bar(x - width, accuracy, width, label='Accuracy')
# Plotting precision
rects2 = ax.bar(x, precision, width, label='Precision')
# Plotting recall
rects3 = ax.bar(x + width, recall, width, label='Recall')
# Plotting F1-score
rects4 = ax.bar(x + 2 * width, f1_score, width, label='F1-score')

# Adding labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Metrics by Class of InceptionResNetV2')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Show plot
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model



# Load your trained model (assuming you've already trained and saved it)
model = load_model('C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/InceptionResNetV2')  # Replace 'your_model_path_here' with the actual path

# Assuming you have the test data ('test_images') and true labels ('test_labels')
# Replace these with your actual test data and labels

# Make predictions on the test data

# Mapping class indices to their respective labels
preds = model.predict(test_x)

# Mapping class indices to their respective labels
class_labels = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Display images with predicted and true labels
plt.figure(figsize=(15, 10))
for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.title('Pred: {}'.format(class_labels[np.argmax(preds[i])], class_labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[24]:


from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Input
IMAGE_SIZE = (224, 224)

# Define ResNet50​ model
DenseNet201 = DenseNet201(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)
for layer in DenseNet201.layers:
    layer.trainable = False
x = Flatten()(DenseNet201.output)
#adding output layer.Softmax classifier is used as it is multi-class classification
prediction = Dense(7, activation='softmax')(x)

model = Model(inputs=DenseNet201.input, outputs=prediction)

# view the structure of the model
print(model.summary())
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)


# In[33]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model
# fit the model
history = model.fit(
  train_x,
  train_y,
  epochs=2,
  callbacks=[early_stop],
  batch_size=32,shuffle=True,validation_data=test_set,validation_steps=len(test_set))


# In[35]:


# accuracies

plt.plot(history.history['accuracy'], label='train acc')

plt.plot(history.history['val_accuracy'], label='val acc')

plt.legend()

plt.savefig('DenseNet201-acc-rps-1.png')

plt.show()


# In[37]:


# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('DenseNet201-rps-1.png')
plt.show()


# In[39]:


score = model.evaluate(test_x, test_y)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[41]:


model.save("C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/DenseNet201")
model.summary()


# In[43]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import seaborn as sns
y_pred_probabilities = model.predict(test_x)
y_pred = np.argmax(y_pred_probabilities, axis=1)  # Convert probabilities to class labels
test_labels  = ['cardboard', 'compost','glass','metal','paper','plastic','trash']
# Create confusion matrix
conf_matrix = confusion_matrix(test_y, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(classification_report(test_y, y_pred, target_names=test_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',xticklabels=test_labels, yticklabels=test_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of DenseNet201')
plt.show()


# In[45]:


# to identify the accuracy of each class 
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)


# Overall accuracy
print ("Cardboard    compost   Glass     Metal     Paper     Plastic    Trash")
ACC = (TP+TN)/(TP+FP+FN+TN)
print (ACC)
Precision=(TP)/(TP+FP) 
print (Precision)
Recall= (TP)/(TP+FN)
print (Recall)
f1_score=(2 * Precision + Recall) / (Precision * Recall)
print (f1_score)


# In[47]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming you have calculated the metrics for each class and stored them in lists or arrays

# List of classes (replace with your class names)
classes = ["Cardboard", "compost", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Example metrics for illustration (replace these with your actual metrics)
accuracy = np.array([0.90425532, 0.9822695,  0.87411348, 0.90780142, 0.82801418, 0.87056738, 0.94503546])
precision = np.array([0.975,      0.85,       0.63768116 ,0.62626263, 0.54901961, 0.69318182, 0.75])
recall = np.array([0.42391304, 0.89473684, 0.48888889, 0.80519481, 0.95726496, 0.57009346, 0.41860465])
f1_score = np.array([0.574358974, 0.341176471, 0.565909091, 0.408064516, 0.391071429, 0.495081967, 0.611111111])

# Plotting the metrics for each class
x = np.arange(len(classes))
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 7))

# Plotting accuracy
rects1 = ax.bar(x - width, accuracy, width, label='Accuracy')
# Plotting precision
rects2 = ax.bar(x, precision, width, label='Precision')
# Plotting recall
rects3 = ax.bar(x + width, recall, width, label='Recall')
# Plotting F1-score
rects4 = ax.bar(x + 2 * width, f1_score, width, label='F1-score')

# Adding labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Metrics by Class of DenseNet201')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Show plot
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()
plt.show()


# In[49]:


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model



# Load your trained model (assuming you've already trained and saved it)
model = load_model('C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/DenseNet201')  # Replace 'your_model_path_here' with the actual path

# Assuming you have the test data ('test_images') and true labels ('test_labels')
# Replace these with your actual test data and labels

# Make predictions on the test data

# Mapping class indices to their respective labels
preds = model.predict(test_x)

# Mapping class indices to their respective labels
class_labels = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Display images with predicted and true labels
plt.figure(figsize=(15, 10))
for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.title('Pred: {}'.format(class_labels[np.argmax(preds[i])], class_labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[51]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
IMAGE_SIZE = (224, 224)

# Define ResNet50 ​ model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in the base ResNet model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer=Adam(lr=0.0001),
  metrics=['accuracy']
)
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model
# fit the model
history = model.fit(
  train_x,
  train_y,
  epochs=2,
  callbacks=[early_stop],
  batch_size=32,shuffle=True,validation_data=test_set,validation_steps=len(test_set))


# In[53]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model
# fit the model
history = model.fit(
  train_x,
  train_y,
  epochs=2,
  callbacks=[early_stop],
  batch_size=32,shuffle=True,validation_data=test_set,validation_steps=len(test_set))


# In[55]:


# accuracies

plt.plot(history.history['accuracy'], label='train acc')

plt.plot(history.history['val_accuracy'], label='val acc')

plt.legend()

plt.savefig('ResNet50-acc-rps-1.png')

plt.show()


# In[56]:


# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('ResNet50-rps-1.png')
plt.show()


# In[57]:


score = model.evaluate(test_x, test_y)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[58]:


model.save("C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/ResNet50")
model.summary()


# In[59]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import seaborn as sns
y_pred_probabilities = model.predict(test_x)
y_pred = np.argmax(y_pred_probabilities, axis=1)  # Convert probabilities to class labels
test_labels  = ['cardboard', 'compost','glass','metal','paper','plastic','trash']
# Create confusion matrix
conf_matrix = confusion_matrix(test_y, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(classification_report(test_y, y_pred, target_names=test_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',xticklabels=test_labels, yticklabels=test_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of ResNet50')
plt.show()


# In[60]:


# to identify the accuracy of each class 
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)


# Overall accuracy
print ("Cardboard    compost   Glass     Metal     Paper     Plastic    Trash")
ACC = (TP+TN)/(TP+FP+FN+TN)
print (ACC)
Precision=(TP)/(TP+FP) 
print (Precision)
Recall= (TP)/(TP+FN)
print (Recall)
f1_score=(2 * Precision + Recall) / (Precision * Recall)
print (f1_score)


# In[61]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming you have calculated the metrics for each class and stored them in lists or arrays

# List of classes (replace with your class names)
classes = ["Cardboard", "compost", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Example metrics for illustration (replace these with your actual metrics)
accuracy = np.array([0.90425532, 0.9822695,  0.87411348, 0.90780142, 0.82801418, 0.87056738, 0.94503546])
precision = np.array([0.975,      0.85,       0.63768116 ,0.62626263, 0.54901961, 0.69318182, 0.75])
recall = np.array([0.42391304, 0.89473684, 0.48888889, 0.80519481, 0.95726496, 0.57009346, 0.41860465])
f1_score = np.array([0.574358974, 0.341176471, 0.565909091, 0.408064516, 0.391071429, 0.495081967, 0.611111111])

# Plotting the metrics for each class
x = np.arange(len(classes))
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 7))

# Plotting accuracy
rects1 = ax.bar(x - width, accuracy, width, label='Accuracy')
# Plotting precision
rects2 = ax.bar(x, precision, width, label='Precision')
# Plotting recall
rects3 = ax.bar(x + width, recall, width, label='Recall')
# Plotting F1-score
rects4 = ax.bar(x + 2 * width, f1_score, width, label='F1-score')

# Adding labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Show plot
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()
plt.show()


# In[62]:


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model



# Load your trained model (assuming you've already trained and saved it)
model = load_model('C:/Users/ashmi/AppData/Local/Programs/Python/Python311/ipcs project/ResNet50')  # Replace 'your_model_path_here' with the actual path

# Assuming you have the test data ('test_images') and true labels ('test_labels')
# Replace these with your actual test data and labels

# Make predictions on the test data

# Mapping class indices to their respective labels
preds = model.predict(test_x)

# Mapping class indices to their respective labels
class_labels = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Display images with predicted and true labels
plt.figure(figsize=(15, 10))
for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.title('Pred: {}'.format(class_labels[np.argmax(preds[i])], class_labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




