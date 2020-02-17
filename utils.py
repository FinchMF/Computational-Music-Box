#################### 
#  Preprocessing   #
####################

# Dependencies Needed

import zipfile
import glob


from pathlib import Path

import glob
import os
import music21
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord, duration, stream

# %matplotlib inline 

# %config InlineBackend.figure_format = 'retina'




def note_to_int(note): # transform pitch name to midi value
    
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if ('#-' in note):
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[3]
        value = base_value + 12*(int(octave)-(-1))
        
    elif ('#' in note): 
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12*(int(octave)-(-1))
        
    elif ('-' in note): 
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12*(int(octave)-(-1))
        
    else:
        first_letter = note[0]
        base_val = note_base_name.index(first_letter)
        octave = note[1]
        value = base_val + 12*(int(octave)-(-1))
        
    return value




# first we need to set our matrix's value 
# rest = (min_value, lower_bound)
# continuation = (lower_bound, upper_bound)
# first_touch = (upper_bound, max_value)

min_value = 0.00
lower_first = 0.0

lower_second = 0.5
upper_first = 0.5

upper_second = 1.0
max_value = 1.0

def notes_to_matrix(notes, durations, offsets, min_value=min_value, lower_first=lower_first,
                    lower_second=lower_second,
                    upper_first=upper_first, upper_second=upper_second,
                    max_value=max_value):
    
    # X axis = time
    # Y axis = pitch values.
    
    # normalize matrix between 0 and 1.
    # a rest will be represented with (min_value, lower_first), 
    # a continuation of a note represetned with [lower_second, upper_first]
    # initiate note represented with (upper_second, max_value)
    
    
    try:
        last_offset = int(offsets[-1]) 
    except IndexError:
        print ('Index Error')
        return (None, None, None)
    
    total_offset_axis = last_offset * 4 + (8 * 4) 
    our_matrix = np.random.uniform(min_value, lower_first, (128, int(total_offset_axis))) 
    # creates matrix and fills with (-1, -0.3), this values will represent the rest.
    
    for (note, duration, offset) in zip(notes, durations, offsets):
        how_many = int(float(duration)/0.25) # indicates time duration for single note.
       
        
        # Define difference between single and double note.
        # I have choose the value for first touch, the another value for continuation.
        # Lets randomize it
        # Uniform distrubition
         
        first_touch = np.random.uniform(upper_second, max_value, 1)
        continuation = np.random.uniform(lower_second, upper_first, 1)
        
        if ('.' not in str(note)): # It is not a chord. Single note.
            our_matrix[note, int(offset * 4)] = first_touch
            our_matrix[note, int((offset * 4) + 1) : int((offset * 4) + how_many)] = continuation

        else: # if it is a chord
            chord_notes_str = [note for note in note.split('.')] 
            chord_notes_float = list(map(int, chord_notes_str)) # Take notes in chord one by one

            for chord_note_float in chord_notes_float:
                our_matrix[chord_note_float, int(offset * 4)] = first_touch
                our_matrix[chord_note_float, int((offset * 4) + 1) : int((offset * 4) + how_many)] = continuation
                
    return our_matrix

def check_float(duration): # function to address the issue which comes from some note's duration. 
                           # Example: some notes have duration like 14/3 or 7/3. 
    if ('/' in duration):
        numerator = float(duration.split('/')[0])
        denominator = float(duration.split('/')[1])
        duration = str(float(numerator/denominator))
    return duration


####################
#  MIDI TO MATRIX  #
####################

def midi_to_matrix(filename, length=250): # Convert midi file to matrix for DL architecture.
    
    midi = music21.converter.parse(filename)
    notes_to_parse = None
    
    try :
        parts = music21.instrument.partitionByInstrument(midi)
    except TypeError:
        print ('Type error.')
        return None
      
    instrument_names = []
    
    try:
        for instrument in parts: # Learn names of instruments.
            name = (str(instrument).split(' ')[-1])[:-1]
            instrument_names.append(name)
    
    except TypeError:
        print ('Type is not iterable.')
        return None
    
    # For now, we want to focus on piano.
    try:
        piano_index = instrument_names.index('Piano')
    except ValueError:
        print ('%s have not any Piano part' %(filename))
        return None
    
    
    notes_to_parse = parts.parts[piano_index].recurse()
    
    duration_piano = float(check_float((str(notes_to_parse._getDuration()).split(' ')[-1])[:-1]))

    durations = []
    notes = []
    offsets = []
    
    for element in notes_to_parse:
        if isinstance(element, note.Note): # If it is single note
            notes.append(note_to_int(str(element.pitch))) # Append note's integer value to "notes" list.
            duration = str(element.duration)[27:-1] 
            durations.append(check_float(duration)) 
            offsets.append(element.offset)

        elif isinstance(element, chord.Chord): # If it is chord
            notes.append('.'.join(str(note_to_int(str(n)))
                                  for n in element.pitches))
            duration = str(element.duration)[27:-1]
            durations.append(check_float(duration))
            offsets.append(element.offset)

    
    
    our_matrix = notes_to_matrix(notes, durations, offsets)
    
    try:
        freq, time = our_matrix.shape
    except AttributeError:
        print ("'tuple' object has no attribute 'shape'")
        return None
            
    if (time >= length):
        return (our_matrix[:,:length]) # set all individual note matrices to same shape for Generative DL.
    else:
        print ('%s have not enough duration' %(filename))


######################
#  BACK TO NOTATION  #
######################  

def int_to_note(integer):
    # Convert midi value to pitchname
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave_detector = (integer // 12) 
    base_name_detector = (integer % 12) 
    note = note_base_name[base_name_detector] + str((int(octave_detector))-1)
    if ('-' in note):
      note = note_base_name[base_name_detector] + str(0)
      return note
    return note





# IMPORTANT_NOTE: Transforming from matrix form to midi form, I need to indicate first touch, continuation and rest with unique numbers.
# -1.0 for rest , 0 for continuation and 1 for first touch.


lower_bound = (lower_first + lower_second) / 2
upper_bound = (upper_first + upper_second) / 2

def converter_func(arr,first_touch = 1.0, continuation = 0.0, lower_bound = lower_bound, upper_bound = upper_bound):
    # First touch represent start for note, continuation represent continuation for first touch, 0 represent end or rest
    np.place(arr, arr < lower_bound, -1.0)
    np.place(arr, (lower_bound <= arr) & (arr < upper_bound), 0.0)
    np.place(arr, arr >= upper_bound, 1.0)
    return arr






def how_many_repetitive_func(array, from_where=0, continuation=0.0):
    new_array = array[from_where:]
    count_repetitive = 1 
    for i in new_array:
        if (i != continuation):
            return (count_repetitive)
        else:
            count_repetitive += 1
    return (count_repetitive)






def matrix_to_midi(matrix, random=0):
    first_touch = 1.0
    continuation = 0.0
    y_axis, x_axis = matrix.shape
    output_notes = []
    offset = 0
        
    # Delete rows until reaching the row which includes 'first_touch'
    how_many_in_start_zeros = 0
    for x_axis_num in range(x_axis):
        one_time_interval = matrix[:,x_axis_num] # Values in a column.
        one_time_interval_norm = converter_func(one_time_interval)
        if (first_touch not in one_time_interval_norm):
            how_many_in_start_zeros += 1
        else:
            break
            
    how_many_in_end_zeros = 0
    for x_axis_num in range(x_axis-1,0,-1):
        one_time_interval = matrix[:,x_axis_num] # values in a column
        one_time_interval_norm = converter_func(one_time_interval)
        if (first_touch not in one_time_interval_norm):
            how_many_in_end_zeros += 1
        else:
            break
        
    print ('How many rows for non-start note at beginning:', how_many_in_start_zeros)
    print ('How many rows for non-start note at end:', how_many_in_end_zeros)

    matrix = matrix[:,how_many_in_start_zeros:]
    y_axis, x_axis = matrix.shape
    print (y_axis, x_axis)

    for y_axis_num in range(y_axis):
        one_freq_interval = matrix[y_axis_num,:] # Values in a row.
        
        one_freq_interval_norm = converter_func(one_freq_interval)
        
        i = 0        
        offset = 0
        
        if (random):
          
          while (i < len(one_freq_interval)):
              how_many_repetitive = 0
              temp_i = i
              if (one_freq_interval_norm[i] == first_touch):
                  how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i+1, continuation=continuation)
                  i += how_many_repetitive 

              if (how_many_repetitive > 0):
                  random_num = np.random.randint(3,6)
                  new_note = note.Note(int_to_note(y_axis_num),duration=duration.Duration(0.25*random_num*how_many_repetitive))
                  new_note.offset = 0.25*temp_i*2
                  new_note.storedInstrument = instrument.Piano()
                  output_notes.append(new_note)
              else:
                  i += 1
        
          
        else:
          
          while (i < len(one_freq_interval)):
              how_many_repetitive = 0
              temp_i = i
              if (one_freq_interval_norm[i] == first_touch):
                  how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i+1, continuation=continuation)
                  i += how_many_repetitive 

              if (how_many_repetitive > 0):
                  new_note = note.Note(int_to_note(y_axis_num),duration=duration.Duration(0.25*how_many_repetitive))
                  new_note.offset = 0.25*temp_i
                  new_note.storedInstrument = instrument.Piano()
                  output_notes.append(new_note)
              else:
                  i += 1
        
    return output_notes



############## 
#  DATABASE  #
##############

# This database is a numpy array file which stores all the midi files transformed into matrices 

#################
# BACH DATABASE #
#################

bach_database_npy = 'bach_midi_files.npy'
bach_file_database_npy = Path('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Bach/midi_files/' + bach_database_npy)


if bach_file_database_npy.is_file(): 
    bach_midis_array = np.load(bach_file_database_npy)
    
else:
    print (os.getcwd())
    # root_dir = ('./content/midi_files') - vestigial component left from google colab. Decided to go more direct with file path
    all_midi_paths = glob.glob(os.path.join('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Bach/midi_files/*mid'))
    print (all_midi_paths)
    matrix_of_all_midis = []

    # All midi have to be in same shape. 

    for single_midi_path in all_midi_paths:
        print (single_midi_path)
        matrix_of_single_midi = midi_to_matrix(single_midi_path, length=300)
        if (matrix_of_single_midi is not None):
            matrix_of_all_midis.append(matrix_of_single_midi)
            print (matrix_of_single_midi.shape)
    bach_midis_array = np.asarray(matrix_of_all_midis)
    np.save(bach_file_database_npy, bach_midis_array)
    
bach_midis_array_raw = bach_midis_array
print ((bach_midis_array_raw.shape))

# Final Matrix Transformation for LSTM input

bach_midis_array = np.transpose(bach_midis_array_raw, (0, 2, 1)) 
bach_midis_array = np.asarray(bach_midis_array)
bach_midis_array.shape



bach_midis_array = np.reshape(bach_midis_array,(-1,128))
bach_midis_array.shape



max_len = 15 # number of columns to predict next column.
step = 1 # step size.

previous_full = []
predicted_full = []

for i in range (0, bach_midis_array.shape[0]-max_len, step):
    prev = bach_midis_array[i:i+max_len,...] # take max_len column.
    pred = bach_midis_array[i+max_len,...] # take (max_len)th column.
    previous_full.append(prev)
    predicted_full.append(pred)


previous_full = np.asarray(previous_full).astype('float64')
predicted_full = np.asarray (predicted_full).astype('float64')


num_of_sample, max_len, freq = previous_full.shape

print (previous_full.shape)
print (predicted_full.shape)



######################
# BEETHOVEN DATABASE #
######################

beethoven_database_npy = 'beethoven_midi_files.npy'
beethoven_file_database_npy = Path('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Beethoven/midi_files/' + beethoven_database_npy)


if beethoven_file_database_npy.is_file(): 
    beethoven_midis_array = np.load(beethoven_file_database_npy)
    
else:
    print (os.getcwd())
    # root_dir = ('./content/midi_files') - vestigial component left from google colab. Decided to go more direct with file path
    all_midi_paths = glob.glob(os.path.join('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Beethoven/midi_files/*mid'))
    print (all_midi_paths)
    matrix_of_all_midis = []

    # All midi have to be in same shape. 

    for single_midi_path in all_midi_paths:
        print (single_midi_path)
        matrix_of_single_midi = midi_to_matrix(single_midi_path, length=300)
        if (matrix_of_single_midi is not None):
            matrix_of_all_midis.append(matrix_of_single_midi)
            print (matrix_of_single_midi.shape)
    beethoven_midis_array = np.asarray(matrix_of_all_midis)
    np.save(beethoven_file_database_npy, beethoven_midis_array)
    
beethoven_midis_array_raw = beethoven_midis_array
print ((beethoven_midis_array_raw.shape))

# Final Matrix Transformation for LSTM input

beethoven_midis_array = np.transpose(beethoven_midis_array_raw, (0, 2, 1)) 
beethoven_midis_array = np.asarray(beethoven_midis_array)
beethoven_midis_array.shape



beethoven_midis_array = np.reshape(beethoven_midis_array,(-1,128))
beethoven_midis_array.shape



max_len = 15 # number of columns to predict next column.
step = 1 # step size.

previous_full = []
predicted_full = []

for i in range (0, beethoven_midis_array.shape[0]-max_len, step):
    prev = beethoven_midis_array[i:i+max_len,...] # take max_len column.
    pred = beethoven_midis_array[i+max_len,...] # take (max_len)th column.
    previous_full.append(prev)
    predicted_full.append(pred)


previous_full = np.asarray(previous_full).astype('float64')
predicted_full = np.asarray (predicted_full).astype('float64')


num_of_sample, max_len, freq = previous_full.shape

print (previous_full.shape)
print (predicted_full.shape)



###################
# CHOPIN DATABASE #
###################

chopin_database_npy = 'chopin_t_midi_files.npy'
chopin_file_database_npy = Path('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Chopin/midi_files/' + chopin_database_npy)


if chopin_file_database_npy.is_file(): 
    chopin_midis_array = np.load(chopin_file_database_npy)
    
else:
    print (os.getcwd())
    # root_dir = ('./content/midi_files') - vestigial component left from google colab. Decided to go more direct with file path
    all_midi_paths = glob.glob(os.path.join('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Chopin/Chopin_Subset/midi_files/*mid'))
    print (all_midi_paths)
    matrix_of_all_midis = []

    # All midi have to be in same shape. 

    for single_midi_path in all_midi_paths:
        print (single_midi_path)
        matrix_of_single_midi = midi_to_matrix(single_midi_path, length=300)
        if (matrix_of_single_midi is not None):
            matrix_of_all_midis.append(matrix_of_single_midi)
            print (matrix_of_single_midi.shape)
    chopin_midis_array = np.asarray(matrix_of_all_midis)
    np.save(chopin_file_database_npy, chopin_midis_array)
    
chopin_midis_array_raw = chopin_midis_array
print ((chopin_midis_array_raw.shape))

# Final Matrix Transformation for LSTM input

chopin_midis_array = np.transpose(chopin_midis_array_raw, (0, 2, 1)) 
chopin_midis_array = np.asarray(chopin_midis_array)
chopin_midis_array.shape



chopin_midis_array = np.reshape(chopin_midis_array,(-1,128))
chopin_midis_array.shape



max_len = 15 # number of columns to predict next column.
step = 1 # step size.

previous_full = []
predicted_full = []

for i in range (0, chopin_midis_array.shape[0]-max_len, step):
    prev = chopin_midis_array[i:i+max_len,...] # take max_len column.
    pred = chopin_midis_array[i+max_len,...] # take (max_len)th column.
    previous_full.append(prev)
    predicted_full.append(pred)


previous_full = np.asarray(previous_full).astype('float64')
predicted_full = np.asarray (predicted_full).astype('float64')


num_of_sample, max_len, freq = previous_full.shape

print (previous_full.shape)
print (predicted_full.shape)



####################
# DEBUSSY DATABASE #
####################

debussy_database_npy = 'debussy_midi_files.npy'
debussy_file_database_npy = Path('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Debussy/midi_files/' + debussy_database_npy)


if debussy_file_database_npy.is_file(): 
    debussy_midis_array = np.load(debussy_file_database_npy)
    
else:
    print (os.getcwd())
    # root_dir = ('./content/midi_files') - vestigial component left from google colab. Decided to go more direct with file path
    all_midi_paths = glob.glob(os.path.join('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Debussy/midi_files/*mid'))
    print (all_midi_paths)
    matrix_of_all_midis = []

    # All midi have to be in same shape. 

    for single_midi_path in all_midi_paths:
        print (single_midi_path)
        matrix_of_single_midi = midi_to_matrix(single_midi_path, length=300)
        if (matrix_of_single_midi is not None):
            matrix_of_all_midis.append(matrix_of_single_midi)
            print (matrix_of_single_midi.shape)
    debussy_midis_array = np.asarray(matrix_of_all_midis)
    np.save(debussy_file_database_npy, debussy_midis_array)
    
debussy_midis_array_raw = debussy_midis_array
print ((debussy_midis_array_raw.shape))

# Final Matrix Transformation for LSTM input

debussy_midis_array = np.transpose(debussy_midis_array_raw, (0, 2, 1)) 
debussy_midis_array = np.asarray(debussy_midis_array)
debussy_midis_array.shape



debussy_midis_array = np.reshape(debussy_midis_array,(-1,128))
debussy_midis_array.shape



max_len = 15 # number of columns to predict next column.
step = 1 # step size.

previous_full = []
predicted_full = []

for i in range (0, debussy_midis_array.shape[0]-max_len, step):
    prev = debussy_midis_array[i:i+max_len,...] # take max_len column.
    pred = debussy_midis_array[i+max_len,...] # take (max_len)th column.
    previous_full.append(prev)
    predicted_full.append(pred)


previous_full = np.asarray(previous_full).astype('float64')
predicted_full = np.asarray (predicted_full).astype('float64')


num_of_sample, max_len, freq = previous_full.shape

print (previous_full.shape)
print (predicted_full.shape)



####################
# RAVEL DATABASE #
####################

ravel_database_npy = 'ravel_midi_files.npy'
ravel_file_database_npy = Path('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Ravel/midi_files/' + ravel_database_npy)


if ravel_file_database_npy.is_file(): 
    ravel_midis_array = np.load(ravel_file_database_npy)
    
else:
    print (os.getcwd())
    # root_dir = ('./content/midi_files') - vestigial component left from google colab. Decided to go more direct with file path
    all_midi_paths = glob.glob(os.path.join('/Volumes/S190813/Music_Box/LTSM_Generative_Music/Ravel/midi_files/*mid'))
    print (all_midi_paths)
    matrix_of_all_midis = []

    # All midi have to be in same shape. 

    for single_midi_path in all_midi_paths:
        print (single_midi_path)
        matrix_of_single_midi = midi_to_matrix(single_midi_path, length=300)
        if (matrix_of_single_midi is not None):
            matrix_of_all_midis.append(matrix_of_single_midi)
            print (matrix_of_single_midi.shape)
    ravel_midis_array = np.asarray(matrix_of_all_midis)
    np.save(ravel_file_database_npy, ravel_midis_array)
    
ravel_midis_array_raw = ravel_midis_array
print ((ravel_midis_array_raw.shape))

# Final Matrix Transformation for LSTM input

ravel_midis_array = np.transpose(ravel_midis_array_raw, (0, 2, 1)) 
ravel_midis_array = np.asarray(ravel_midis_array)
ravel_midis_array.shape



ravel_midis_array = np.reshape(ravel_midis_array,(-1,128))
ravel_midis_array.shape



max_len = 15 # number of columns to predict next column.
step = 1 # step size.

previous_full = []
predicted_full = []

for i in range (0, ravel_midis_array.shape[0]-max_len, step):
    prev = ravel_midis_array[i:i+max_len,...] # take max_len column.
    pred = ravel_midis_array[i+max_len,...] # take (max_len)th column.
    previous_full.append(prev)
    predicted_full.append(pred)


previous_full = np.asarray(previous_full).astype('float64')
predicted_full = np.asarray (predicted_full).astype('float64')


num_of_sample, max_len, freq = previous_full.shape

print (previous_full.shape)
print (predicted_full.shape)


###################
# LSTM MODEL LOAD #
###################

# Dependencies Needed

from keras import layers
from keras import models
import keras
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
from keras.layers.advanced_activations import *

################### 
#  COMPOSER BOTS  #
###################


bach_bot = '/Volumes/S190813/Music_Box/INCOMING/z200120/bach_200120_model.h5'
beethoven_bot = '/Volumes/S190813/Music_Box/INCOMING/BEV_200121/beethoven_model.h5' 
chopin_bot = '/Volumes/S190813/Music_Box/INCOMING/CH_t_200120/chopin_t_model.h5'
debussy_bot = '/Volumes/S190813/Music_Box/INCOMING/DE_200120/debussy_model.h5'
ravel_bot = '/Volumes/S190813/Music_Box/INCOMING/RA_200120/ravel_model.h5'


bach_lstm = load_model(bach_bot)
beethoven_lstm = load_model(beethoven_bot)
chopin_lstm = load_model(chopin_bot)
debussy_lstm = load_model(debussy_bot)
ravel_lstm = load_model(ravel_bot) 



################### 
# GENERATE OUTPUT #  
################### 



def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    num_of_top = 15
    num_of_first = np.random.randint(1,3)

    
    preds [0:48] = 0 # eliminate notes with low octaves
    preds [100:] = 0 # eliminate notes with very high octaves
    
    ind = np.argpartition(preds, -1*num_of_top)[-1*num_of_top:]
    top_indices_sorted = ind[np.argsort(preds[ind])]
    
    
    array = np.random.uniform(0.0, 0.0, (128)) 
    array[top_indices_sorted[0:num_of_first]] = 1.0
    array[top_indices_sorted[num_of_first:num_of_first+3]] = 0.5

    return array




import random

###############
# BACH OUTPUT #
###############


start_index = random.randint(0, len(bach_midis_array)- max_len - 1)
    
generated_midi = bach_midis_array[start_index: start_index + max_len]


for temperature in [1.2, 1.6]:
        print('------ temperature:', temperature)
        generated_midi = bach_midis_array[start_index: start_index + max_len]
        for i in range(680):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = bach_lstm.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = sample(preds, temperature)
           
            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)
            

        generated_midi_final = np.transpose(generated_midi,(1,0))
        output_notes = matrix_to_midi(generated_midi_final, random=1)
        midi_stream = stream.Stream(output_notes)
        midi_file_name = ('bach_lstm_out_{}.mid'.format(temperature))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, instrument.Piano())
        parsed.write('midi', fp=midi_file_name)



####################
# BEETHOVEN OUTPUT #
####################


start_index = random.randint(0, len(beethoven_midis_array)- max_len - 1)
    
generated_midi = beethoven_midis_array[start_index: start_index + max_len]


for temperature in [1.2, 1.6]:
        print('------ temperature:', temperature)
        generated_midi = beethoven_midis_array[start_index: start_index + max_len]
        for i in range(680):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = beethoven_lstm.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = sample(preds, temperature)
           
            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)
            

        generated_midi_final = np.transpose(generated_midi,(1,0))
        output_notes = matrix_to_midi(generated_midi_final, random=1)
        midi_stream = stream.Stream(output_notes)
        midi_file_name = ('beethoven_lstm_out_{}.mid'.format(temperature))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, instrument.Piano())
        parsed.write('midi', fp=midi_file_name)




#################
# CHOPIN OUTPUT #
#################


start_index = random.randint(0, len(chopin_midis_array)- max_len - 1)
    
generated_midi = chopin_midis_array[start_index: start_index + max_len]


for temperature in [1.2, 1.6]:
        print('------ temperature:', temperature)
        generated_midi = chopin_midis_array[start_index: start_index + max_len]
        for i in range(680):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = chopin_lstm.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = sample(preds, temperature)
           
            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)
            

        generated_midi_final = np.transpose(generated_midi,(1,0))
        output_notes = matrix_to_midi(generated_midi_final, random=1)
        midi_stream = stream.Stream(output_notes)
        midi_file_name = ('chopin_lstm_out_{}.mid'.format(temperature))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, instrument.Piano())
        parsed.write('midi', fp=midi_file_name)




##################
# DEBUSSY OUTPUT #
##################


start_index = random.randint(0, len(debussy_midis_array)- max_len - 1)
    
generated_midi = debussy_midis_array[start_index: start_index + max_len]


for temperature in [1.2, 1.6]:
        print('------ temperature:', temperature)
        generated_midi = debussy_midis_array[start_index: start_index + max_len]
        for i in range(680):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = debussy_lstm.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = sample(preds, temperature)
           
            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)
            

        generated_midi_final = np.transpose(generated_midi,(1,0))
        output_notes = matrix_to_midi(generated_midi_final, random=1)
        midi_stream = stream.Stream(output_notes)
        midi_file_name = ('debussy_lstm_out_{}.mid'.format(temperature))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, instrument.Piano())
        parsed.write('midi', fp=midi_file_name)



################
# RAVEL OUTPUT #
################


start_index = random.randint(0, len(ravel_midis_array)- max_len - 1)
    
generated_midi = ravel_midis_array[start_index: start_index + max_len]


for temperature in [1.2, 1.6]:
        print('------ temperature:', temperature)
        generated_midi = ravel_midis_array[start_index: start_index + max_len]
        for i in range(680):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = ravel_lstm.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = sample(preds, temperature)
           
            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)
            

        generated_midi_final = np.transpose(generated_midi,(1,0))
        output_notes = matrix_to_midi(generated_midi_final, random=1)
        midi_stream = stream.Stream(output_notes)
        midi_file_name = ('ravel_lstm_out_{}.mid'.format(temperature))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, instrument.Piano())
        parsed.write('midi', fp=midi_file_name)        




####################
#  CLASSIFICATION  #
####################


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn import svm, linear_model, naive_bayes, neural_network, neighbors, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import random, math
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from collections import Counter
from itertools import combinations

import itertools

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import warnings
warnings.simplefilter("ignore")
from tqdm import tqdm_notebook as tqdm


