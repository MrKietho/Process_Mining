import pandas as pd
import numpy as np
import datetime, time
import tensorflow as tf
import csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

# Written by Kiet Bennema ten Brinke (TU Eindhoven)
# Inspired by Niek, T., et al.: Predictive business process monitoring with lstm neural networks. In: Advanced Information Systems Engineering. pp. 477{492. Springer (2017)

class LSTM_predictor:
    
    def __init__(self, epochs=1, neurons_per_layer=75):
        self.epochs = epochs
        self.neurons_per_layer = neurons_per_layer

    def preprocess_and_store(self, data, split_index):
        print('\nBuilding extra features...')
        data['timeDelta'] = data['event time:timestamp'].diff().dt.total_seconds()
        data.loc[data['case concept:name'] != data['case concept:name'].shift(1), 'timeDelta'] = 0
        data['endOfTrace'] = 0
        data.loc[data['case concept:name'] != data['case concept:name'].shift(-1), 'endOfTrace'] = 1
        data['timeWithinDay'] = ((data['event time:timestamp'] - data['event time:timestamp'].dt.normalize()) / pd.Timedelta('1 second'))
        data['timeWithinWeek'] = data['event time:timestamp'].apply(lambda date: (date - (date - pd.Timedelta(days=date.dayofweek, hours=date.hour, minutes=date.minute, seconds=date.second, microseconds=date.microsecond))).total_seconds())

        self.max_trace_length = max(data.groupby('case concept:name').size())
        self.timedelta_mean = data['timeDelta'].iloc[:split_index].mean()
        self.test_timedelta_mean = data['timeDelta'].iloc[-(len(data)-split_index):].mean()

        features = pd.get_dummies(data['event concept:name'])
        self.different_events_count = features.shape[1]
        features['endOfTrace'] = data['endOfTrace']
        self.event_names = list(features.columns.values)
        features['timeDelta'] = data['timeDelta'] / self.timedelta_mean
        features['timeWithinDay'] = data['timeWithinDay'] / (60 * 60 * 24)
        features['timeWithinWeek'] = data['timeWithinWeek'] / (60 * 60 * 24 * 7)
        features.reset_index(inplace=True)

        cases = np.split(features.values.tolist(), (features.index[features['endOfTrace'] == 1] + 1)[:-1].tolist())
        inputs = []

        print('\nConstructing sequential inputs...')

        # padding = [0] * (self.different_events_count + 4)
        for i, case in enumerate(cases):    # first add only the the first event as input, then the first two, then first three etc.
            cases[i] = cases[i].tolist()
            length = len(case)
            if length > 0:   
                for k in range(0, length):
                    cases[i][k].pop(0)      # remove unwanted index in array from to_numpy()
                # cases[i] = np.delete(cases[i], 0, 1)
                for j in range(1, length + 1):
                    pad_length = self.max_trace_length - j
                    partial_trace = ([[0] * (self.different_events_count + 4)] * pad_length) + cases[i][0:j]  # add padding to the left of input, as input size needs to be fixed 
                    # partial_trace = np.concatenate(([[0] * (self.different_events_count + 4)] * pad_length, cases[i][0:j]))
                    # for _ in range(pad_length):
                    #     partial_trace.insert(0, padding)    # add padding to the left of input, as input size needs to be fixed
                    inputs.append(partial_trace)

        # validation_range = 0.2
        # for i in range(int((1-validation_range)*split_index), int((1-validation_range)*split_index) + self.max_trace_length):
        #     if inputs[i][-2] == padding:
        #       validation_split_index = i
        # self.validation_split_index = validation_split_index
        # train_cases = inputs[:validation_split_index-1] + inputs[validation_split_index:split_index-1]
        train_cases = inputs[:split_index-1]
        # validation_inputs = inputs[validation_split_index:split_index-1]
        test_cases = inputs[-(len(inputs)-split_index):]

        print('\nConstructing numpy arrays...')
        self.inputs = np.array(train_cases)
        # validation_inputs = np.array(validation_inputs, dtype=np.float32)
        self.test_inputs = np.array(test_cases)

        print("\nConstruction done. ")

        training_targets = pd.get_dummies(data['event concept:name'])[:split_index]
        self.training_targets_events = training_targets.shift(-1).values[:-1]
        self.training_targets_times = data[:split_index]['timeDelta'].shift(-1).values[:-1] / self.timedelta_mean
        
        # validation_targets_events = pd.get_dummies(data['event concept:name']).shift(-1).values[validation_split_index:split_index-1]
        # validation_targets_times = data[validation_split_index:split_index]['timeDelta'].shift(-1).values[:-1] / self.timedelta_mean
 
        # self.validation_inputs = tf.data.Dataset.from_tensors((validation_inputs, (validation_targets_events, validation_targets_times)))

    
    def fit(self):
        main_input = tf.keras.Input(shape=(self.max_trace_length, self.different_events_count + 4))
        layer1 = tf.keras.layers.GRU(self.neurons_per_layer, return_sequences=True)(main_input)
        b1 = tf.keras.layers.BatchNormalization()(layer1)
        layer_2_1 = tf.keras.layers.GRU(self.neurons_per_layer, return_sequences=False)(b1)
        layer_2_2 = tf.keras.layers.GRU(self.neurons_per_layer, return_sequences=False)(b1)
        b2_1 = tf.keras.layers.BatchNormalization()(layer_2_1)
        b2_2 = tf.keras.layers.BatchNormalization()(layer_2_2)
        event_output = tf.keras.layers.Dense(self.different_events_count, activation='softmax', name='event_output')(b2_1)
        time_output = tf.keras.layers.Dense(1, name='time_output')(b2_2)

        self.model = tf.keras.Model(inputs=[main_input], outputs=[event_output, time_output])

        opt = tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
        
        self.model.compile(loss={'event_output':'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint.ckpt', save_weights_only=True, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=2, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        
        self.model.summary()
        self.model.fit(self.inputs, {'event_output':self.training_targets_events, 'time_output':self.training_targets_times}, validation_split=0.2, 
                       epochs=self.epochs, batch_size=self.max_trace_length, callbacks=[checkpoint, early_stopping, lr_reducer])

    
    def predict(self):
        predicted_events = []
        predicted_timestamps = []

        test_output = self.model.predict(self.test_inputs, batch_size=self.max_trace_length)
        self.test_output = pd.DataFrame(test_output[0])
        
        for trace in test_output[0]:
            predicted_events.append(self.event_names[np.argmax(trace)])

        for trace in test_output[1]:
            if trace[0] > 0:
                predicted_timestamps.append(pd.Timedelta(seconds=(trace[0] * self.test_timedelta_mean)))
            else:
                predicted_timestamps.append(pd.Timedelta(seconds=0))

        self.timedeltas = pd.Series(predicted_timestamps)
        self.predicted_events = pd.Series(predicted_events)
        self.predicted_timestamps = test_data['event time:timestamp'].reset_index(drop=True) + self.timedeltas.reset_index(drop=True)


    def event_accuracy(self):   #TODO: Ignore end of trace predictions
        n = test_data['event concept:name'].shape[0]
        error = test_data['event concept:name'].reset_index(drop=True) == self.predicted_events.shift(1).reset_index(drop=True)
        return error.sum() / n

    def timestamp_accuracy(self):
        n = test_data['event concept:name'].shape[0]
        error = abs(test_data['event time:timestamp'].reset_index(drop=True) - self.predicted_timestamps.shift(1).reset_index(drop=True))
        return error.mean()


def read_csv(filepath):
    data = {'eventID': [], 'event concept:name': [], 'case concept:name': [], 'event time:timestamp': []}

    training_input_file = csv.reader(open(filepath))
    line_counter = 0
    replace_application = False
    for line in training_input_file:
        if 'BPI_Challenge_2017' in filepath and line_counter > 0 and line_counter < DATA_ROWS:
            data['eventID'].append(line[0])
            data['event concept:name'].append(line[7])
            data['case concept:name'].append(line[3])
            data['event time:timestamp'].append(line[11])
            replace_application = True
        elif 'BPI_Challenge_2012' in filepath and line_counter > 0 and line_counter < DATA_ROWS:
            data['eventID'].append(line[0])
            data['event concept:name'].append(line[4])
            data['case concept:name'].append(line[1])
            data['event time:timestamp'].append(line[6])
            replace_application = True
        elif 'BPI_Challenge_2018' in filepath and line_counter > 0 and line_counter < DATA_ROWS:
            data['eventID'].append(line[0])
            data['event concept:name'].append(line[3])
            data['case concept:name'].append(line[1])
            data['event time:timestamp'].append(line[2])
        elif 'Road_Traffic_Fine_Management_Proc' in filepath and line_counter > 0 and line_counter < DATA_ROWS:
            data['eventID'].append(line[0])
            data['event concept:name'].append(line[2])
            data['case concept:name'].append(line[1])
            data['event time:timestamp'].append(line[4])
      
        line_counter += 1

    print('\nDone reading')

    data = pd.DataFrame.from_dict(data)
    data['event time:timestamp'] = pd.to_datetime(data['event time:timestamp'], dayfirst=True)
    if replace_application:
        data['case concept:name'] = data['case concept:name'].str.replace("Application_", "").astype('int32')

    data = data.sort_values('event time:timestamp')

    train_data = data[:int(0.65*len(data))]
    test_data = data[-int(0.35*len(data)):]

    cond = train_data['case concept:name'].isin(test_data['case concept:name'])
    cond2 = test_data['case concept:name'].isin(train_data['case concept:name'])
    train_data.drop(train_data[cond].index, inplace=True)
    test_data.drop(test_data[cond2].index, inplace=True)

    split_case = test_data.iloc[0]['case concept:name']
    test_data.drop(test_data[test_data['case concept:name'] == split_case].index, inplace=True)

    train_data = train_data.sort_values('case concept:name', kind='mergesort')
    test_data = test_data.sort_values('case concept:name', kind='mergesort')

    return (train_data, test_data)


#
# Hyperparameters, filepath and amount of data to load
#

FILEPATH = 'Data/BPI_Challenge_2012.csv'
NEURONS_PER_LAYER = 100
EPOCHS = 60
DATA_ROWS = 5000


old_time = time.time()

print('\nReading, splitting and storing data...')
train_data, test_data = read_csv(FILEPATH)
data = pd.concat([train_data, test_data])
print('\nDone. Took {:2f} seconds'.format(time.time() - old_time))

predictor = LSTM_predictor(neurons_per_layer=NEURONS_PER_LAYER, epochs=EPOCHS)

old_time = time.time()
if not os.path.isfile('data.h5'):
    print('\nPreprocessing {} training entries and {} test entries...'.format(len(train_data), len(test_data)))
    predictor.preprocess_and_store(data, train_data.shape[0])

print('\nDone. Time spent: {}s'.format(time.time() - old_time))

print('\nFitting the model using {} epochs...'.format(predictor.epochs))
# inputs = predictor.inputs
# test_inputs = predictor.test_inputs
# targets = predictor.training_targets_events
# targets_time = predictor.training_targets_times

old_time = time.time()
predictor.fit()
print('\nDone. Time spent: {}s'.format(time.time() - old_time))
# test = predictor.test_inputs
print('\nPredicting...')
predictor.predict()
# test_inputs = predictor.test_inputs
# test_outputs = predictor.test_output
# timedeltas = predictor.timedeltas
# predictions_events = predictor.predicted_events
# prediction_timestamps = predictor.predicted_timestamps

print('\nEvent Accuracy: {:.2f}%'.format(predictor.event_accuracy() * 100))
print('\nTimestamp Accuracy: {}'.format(str(predictor.timestamp_accuracy())))
print('\nTimedelta mean: {}'.format(predictor.timedelta_mean))
# print('\nA couple of predicted events:\n')
# print(predictor.predicted_events.shift(1).head(100).values)
#validation_split_index=predictor.validation_split_index
