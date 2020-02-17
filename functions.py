
import music21
import pickle
import numpy as np


# instance of singular parse and notes save
file = "/Volumes/S_4/Music_Box/Scraped_midi_data/Chopin/midi_files/ballade1.mid"
midi = converter.parse(file)
notes_to_parse = midi.flat.notes
notes_small = []
for element in notes_to_parse:
    print(element)
    if isinstance(element, note.Note):
        notes_small.append(str(element.pitch))
    elif isinstance(element, chord.Chord):
        notes_small.append('.'.join(str(n) for n in element.normalOrder))
with open('notes', 'wb') as filepath:
    pickle.dump(notes_small, filepath)