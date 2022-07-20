# Load Dependencies
from tensorflow import keras
import numpy as np
from keras.preprocessing import sequence #ده موديول مختص بالتعامل مع السيكونس داتا او الداتا المتعلقة بالتايم
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN , Flatten
from keras.datasets import imdb
from keras.callbacks import TensorBoard #خاص بعرض البيانات علي سكرين تينسور بورد

# Hyper-Parameters
max_features = 5000
no_classes = 1 #علشان عندي تصنيف باينري و ماعملتش 2 ليه هي هي خلي بالك لانك كده بتقول ان اي فيلم مثلا لو كويس ف هو معانا غير كده ف هو خلاص مش مناسب ف كأنك عملتها تو 
#و في الاحالة دي مش هيبقي الاوتبوت بتاعك سوفت ماكس هيبقي سيجمويد
max_length = 100 #لو فيه كومنتات اقل من 100 هيعمل باد سيكونس علشان لو اقل يعمل بادينج يكملهم 100 لو اكتر هيحذف 
batch_size = 64 #كام جملة في المرة و لو طيته ب نن تقدر تدخل اي حجم من الداتا 
embedding_size = 64 #كام جملة في المرة
dropout_rate = 0.5
hidden_layer_size = 250
no_epochs = 5

# Load IMDB Data from Keras datasets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print('Data loaded successfully.')
print('# Train Data = ', len(x_train))
print('# Test Data = ', len(x_test))

# Data Preprocessing
print('Preprocessing Data..')
x_train = sequence.pad_sequences(x_train, maxlen=max_length) #سيكونس من الموديول بري بروسيسنج من كيراس و هو هنا عمل البادينج 
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

# Design Neural Network Architecture with SimpleRNN
print('Building Simple RNN Model..')

RNN_model = Sequential()
# Add Embedding layer
RNN_model.add(Embedding(max_features, embedding_size, input_length=max_length))
RNN_model.add(Dropout(dropout_rate))
# Add Simple RNN layer
RNN_model.add(SimpleRNN(40,return_sequences = True, batch_input_shape=(1, 3))) 
# Add Dense Hidden Layer
RNN_model.add(Dense(hidden_layer_size, activation='relu')) #ال ار ان ان نفسها مفيهاش هيدن لايرز و كأنك هنا بتعمل ديب نيورال نتوورك 
RNN_model.add(Flatten())
RNN_model.add(Dropout(dropout_rate)) #مش شرط تعما دروب اوت بنفس الريت كل مرة 
# Output Layer 
RNN_model.add(Dense(no_classes, activation='sigmoid'))

# Configure model
RNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# TensorBoard
tensorboard = TensorBoard('./logs/SimpleRNN')

# Train!
print('Training the model..')
RNN_model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=no_epochs, validation_data=(x_test, y_test), callbacks = [tensorboard])


