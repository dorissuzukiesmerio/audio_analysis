import pandas
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

# on cmd puyhon -m pip keras
import keras
from keras import layers
from keras.models import Sequential 


# DEEP LEARNING 

dataset = pandas.read_csv("dataset.csv")
print(dataset)

dataset = pandas.drop(['filename'], axis = 1) # Attention: important to have filename as a column, but not for this analysis

target = dataset.iloc[:,-1] # the genre (which is categorical)
encoder = LabelEncoder()
target = encoder.fit_transform(target)

data = dataset.iloc[:,:-1] # everything except the last column
# z score need to 
scaler = StandardScaler()
data = scaler.fit_transform(numpy.array(data, dtype = float)) # the float is to make the program run faster

print(target)
print(data)

# TRY DOING K-FOLD, AND RANDOM FOREST

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=0)


machine = Sequential() # add the layers one by one, includign the input and output layers
machine.add(layers.Dense(256, activation='relu', input_shape=data_training.shape[1],)) # Remember explanation about Deep learning and layers
machine.add(layers.Dense(128, activation='relu')) # Remember explanation about Deep learning and layers
machine.add(layers.Dense(64, activation='relu')) # Remember explanation about Deep learning and layers
machine.add(layers.Dense(3, activation='softmax')) # Remember explanation about Deep learning and layers
machine.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy']) # this is not the same metrics of accuracy score we had before
# Relu is : if it is >0 set =1, <0 set =0 # CONFERIR 

machine.fit(data_training, target_training, epochs=100, batch_size=128 ) # epoch is the back propagation
# 100 affects accuracy. If your accuracy is low, then you can adjust 
# ABOVE 95 %

# different from the data_training target_training => we are still here ! the pre propagation
# not to the traditional machine learning with data_test and target_test 


new_target = numpy.argmax(machine.predict(data_test), axis = 1) 
print(new_target)
print(metrics.accuracy_score(new_target, target_test)) 

# Accuracy is 100 % ! 

