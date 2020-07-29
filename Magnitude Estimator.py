#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py


waveforms_filepath = '../waveforms.hdf5'
metadata_filepath = '../metadata.csv'




def data_reader(waveforms_filepath, trace_List):
    '''
    For each trace, this function appends a waveform into X and it's corresponding magnitude in y.
    Returns X as a 3D array and y as a 1D array
    '''

    hdf = h5py.File(waveforms_filepath, 'r')
    X = np.zeros([len(trace_List), 3000, 3])
    Y = np.zeros([len(trace_List), 1])

    for i, trace in enumerate(trace_List):
        dataset = hdf.get('earthquake/local/'+str(trace))
        data = np.array(dataset)
        p_arrival = int(dataset.attrs['p_arrival_sample'])
        magnitude = round(float(dataset.attrs['source_magnitude']), 2)

        cropped_data = data[p_arrival-100: p_arrival+2900, :] #30 secs
        X[i,:,:] = cropped_data
        Y[i,0] = magnitude

    hdf.close()

    return X, Y




def string_to_snr(string):
    '''
    This function takes as input a string with 3 space seperated values
    for SNR and converts them into float
    '''

    tempList = string.split()
    snr = []

    for counter, item in enumerate(tempList):
        '''
        Some are like '[56.79999924 55.40000153 47.40000153]'
        and some are '[ 56.79999924 55.40000153 47.40000153 ]'
        '''
        if item != '[' and item != ']':
            left = item.split('[')
            right = item.split(']')

            #if dealing with left most value
            if len(left) == 2:
                value = left[1]
            #if dealing with right most valie
            elif len(right) == 2:
                value = right[0]
            #if dealing with middle value
            elif len(left) == 1 and len(right) == 1:
                value = item
            else:
                pass
            try:
                value = float(value)
            except:
                value = None

            snr.append(value)

    return snr



df = pd.read_csv(metadata_filepath)

def reduce_size(df):
    '''
    Simply reducing the size of df and only fetching waveforms that start and end within 30 secs
    '''
    df = df[df['trace_category'] == 'earthquake_local']
    df = df[df['source_distance_km'] <= 20] #distance between epicenter and receiving station
    df = df[df['source_magnitude_type'] == 'ml'] #richter scale(local magnitude)
    df = df[df['p_arrival_sample'] >= 200] #all waveforms with p waves arrival at 2secs or onwards
    df = df[df['p_arrival_sample']+2900 <= 6000] #starts 1 sec before P wave arrival so 29 secs left after it for a total of 30secs
    df = df[df['coda_end_sample'] <= 3000] #at most 3000 sample
    df = df[df['p_travel_sec'].notnull()]
    df = df[df['p_travel_sec'] > 0]
    df = df[df['source_distance_km'].notnull()]
    df = df[df['source_distance_km'] > 0]
    df = df[df['source_depth_km'].notnull()]
    df = df[df['source_magnitude'].notnull()]
    #Calculating mean of snr values and replacing the values with mean in df
    df['snr_db'] = df['snr_db'].apply(lambda x: np.mean(string_to_snr(x)))
    df = df[df['snr_db'] >= 20]
    return df

df = reduce_size(df)

unique_receiver_codes = df['receiver_code'].unique()


print(len(unique_receiver_codes))

#Finding Station receiver codes with 400 or more observations recorded
def required_stations(codes):
    multi_observations = []
    for i in range(0, len(codes)):
        count = sum(n == str(codes[i]) for n in df['receiver_code'])
        if count >= 400:
            multi_observations.append(codes[i])
    return multi_observations

mutli_observations = required_stations(unique_receiver_codes)



print(len(multi_observations))

np.save('assets/multi_observations', multi_observations)


multi_observations = np.load('assets/multi_observations.npy')

#Station receiver codes with 400 or more observations recorded
print(multi_observations)
print(len(multi_observations))


def get_traces(multi_observations):
    '''
    Returns a list of traces corresponding to station codes in multi_observations
    '''
    trace_List = []
    for i in range(0,len(df)):
        code = df.iloc[i]['receiver_code']
        if code in multi_observations:
            trace_List.append(df.iloc[i]['trace_name'])
    return trace_List

trace_List = get_traces(multi_observations)

print(len(trace_List))


x = df['source_magnitude'].to_list()
plt.hist(x, bins=30, histtype='bar', alpha=0.7, ec='black')
plt.title('Max: ' + str(max(df['source_magnitude'])) + ' ---- Min: ' + str(min(df['source_magnitude'])))
plt.xlabel('Earthquake Magnitude',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.show()


#shuffling trace list
trace_List = np.array(trace_List)
np.random.shuffle(trace_List)

# first 70% data
training = trace_List[: int(0.7*len(trace_List))]

# between 70 and 80%
validation = trace_List[int(0.7*len(trace_List)) : int(0.8*len(trace_List))]

#last 20% data
test = trace_List[int(0.8*len(trace_List)) :]


print(len(training))
print(len(test))

X_train, y_train = data_reader(waveforms_filepath, training)
X_test, y_test = data_reader(waveforms_filepath, test)


X_train.shape, X_test.shape

y_train.shape, y_test.shape

assert not np.any(np.isnan(X_train).any())
assert not np.any(np.isnan(X_test).any())
assert not np.any(np.isnan(y_train).any())
assert not np.any(np.isnan(y_test).any())


from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


Conv1D()
filters = [8, 16, 32, 64]

inp = Input(shape=(3000, 3), name='input_layer')

e = Conv1D(filters[1], 3, padding = 'same')(inp)
e = Dropout(0.3)(e, training=True)
e = MaxPooling1D(4, padding='same')(e)

e = Conv1D(filters[0], 3, padding = 'same')(e)
e = Dropout(0.3)(e, training=True)
e = MaxPooling1D(4, padding='same')(e)

e = Bidirectional(LSTM(10, return_sequences=False, dropout=0.0, recurrent_dropout=0.0))(e)

e = Dense(2)(e)
o = Activation('linear', name='output_layer')(e)

model = Model(inputs=[inp], outputs=o)


from keras import backend as K

def customLoss(yTrue, yPred):
    '''
    minimizing the mean square error
    '''
    y_hat = K.reshape(yPred[:, 0], [-1, 1])
    s = K.reshape(yPred[:, 1], [-1, 1])
    return tf.reduce_sum(0.5 * K.exp(-1 * s) * K.square(K.abs(yTrue - y_hat)) + 0.5 * s, axis=1)

model.compile(optimizer=Adam(learning_rate = 0.01), loss=customLoss)

history = model.fit(X_train, y_train, epochs=40, validation_data= (X_test, y_test), batch_size=64, verbose=1)

np.save('assets/X_train', X_train)


np.save('assets/y_train', y_train)
np.save('assets/X_test', X_test)
np.save('assets/y_test', y_test)

X_val, y_val = data_reader(waveforms_filepath, validation)

np.save('assets/X_val', X_val)
np.save('assets/y_val', y_val)

prediction = model.predict(X_val)[:,0]

y_val.reshape(-1)
prediction.reshape(-1)

model.save('models/estimator_4_17_pm.h5')

df = pd.DataFrame(columns=['actual'], data = y_val)

df['predicted'] = prediction

df['diff'] = abs(df['predicted'] - df['actual'])

df.to_csv('test_results.csv', index=False)

sum(df['diff'])/len(df) #average error of +- 0.32
