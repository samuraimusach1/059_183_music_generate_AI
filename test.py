import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pretty_midi
import librosa.display
from midi2audio import FluidSynth
from tqdm import tqdm
import random
import pygame
import os
import torch
from collections import OrderedDict
from torchsummary import summary
from GPT2RGAX import GPT, GPTConfig
import TMIDIX
from IPython.display import Audio
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

class MiniMuseMusicGenerator:
    def __init__(self, model_path, training_data_path):
        self.load_model(model_path)
        self.load_training_data(training_data_path)
        self.create_ui()

    def load_model(self, model_path):
        print('Loading the model...')
        config = GPTConfig(512,
                           1024,
                           dim_feedforward=1024,
                           n_layer=16,
                           n_head=16,
                           n_embd=1024,
                           enable_rpr=True,
                           er_len=1024)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GPT(config)

        state_dict = torch.load(model_path, map_location=device)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module'
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)

        self.model.to(device)

        self.model.eval()

        print('Model loaded successfully!')
        summary(self.model)

    def load_training_data(self, training_data_path):
        print('Loading training data...')
        self.train_data = TMIDIX.Tegridy_Any_Pickle_File_Reader(training_data_path)
        print('Training data loaded successfully!')

    def generate_music(self):
        number_of_prime_tokens = 256

        instruments = []

        if self.instruments['Piano'].get():
            instruments += [0]

        if self.instruments['Guitar'].get():
            instruments += [1]

        if self.instruments['Bass'].get():
            instruments += [2]

        if self.instruments['Violin'].get():
            instruments += [3]

        if self.instruments['Cello'].get():
            instruments += [4]

        if self.instruments['Harp'].get():
            instruments += [5]

        if self.instruments['Trumpet'].get():
            instruments += [6]

        if self.instruments['Clarinet'].get():
            instruments += [7]

        if self.instruments['Flute'].get():
            instruments += [8]

        if self.instruments['Drums'].get():
            instruments += [9]

        if self.instruments['Choir'].get():
            instruments += [10]

        iidx = []

        print('Looking for matching primes...')

        for i in tqdm(range(0, len(self.train_data), 1025)):
            instr = sorted(list(set([y // 10 for y in self.train_data[i+1:i+1025:4][:number_of_prime_tokens]])))
            if instr == sorted(instruments):
                iidx.append(i)

        print('Found', len(iidx), 'matching primes...')

        iindex = random.choice(iidx)
        print('Selected prime #', iindex // 1025)
        print('Prime index:', iindex)

        out1 = self.train_data[iindex:iindex+number_of_prime_tokens+1]

        if len(out1) != 0:

            song = out1
            song_f = []
            time = 0
            dur = 0
            vel = 0
            pitch = 0
            channel = 0

            son = []

            song1 = []

            for s in song:
                if s > 127:
                    son.append(s)

                else:
                    if len(son) == 4:
                        song1.append(son)
                    son = []
                    son.append(s)

            for s in song1:

                channel = s[0] // 10

                vel = (s[0] % 10) * 16

                time += (s[1]-128) * 16

                dur = (s[2] - 256) * 32

                pitch = (s[3] - 384)

                song_f.append(['note', time, dur, channel, pitch, vel])

            detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                                   output_signature='Mini Muse',
                                                                   output_file_name='/content/Mini-Muse-Music-Composition',
                                                                   track_name='Project Los Angeles',
                                                                   list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 0, 0, 0, 0, 0],
                                                                   number_of_ticks_per_quarter=500)

            print('Done!')

        print('Displaying resulting composition...')
        fname = '/content/Mini-Muse-Music-Composition'

        pm = pretty_midi.PrettyMIDI(fname + '.mid')
        pygame.mixer.init()

        print("pm = " , pm)
        # Retrieve piano roll of the MIDI file
        piano_roll = pm.get_piano_roll()

        plt.figure(figsize=(14, 5))
        librosa.display.specshow(piano_roll, x_axis='time', y_axis='cqt_note', fmin=1, hop_length=160, sr=16000, cmap=plt.cm.hot)
        plt.title(fname)

        midi_data = pm.synthesize()
        audio_data = midi_data.T

        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.mixer.music.set_volume(1.0)

        
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()

        
        pygame.event.wait()

        pygame.mixer.quit()

        print('Done!')

    def create_ui(self):
        self.root = tk.Tk()
        self.root.title("Mini Muse Music Generator")

        ttk.Label(self.root, text="Number of Prime Tokens").grid(row=0, column=0, padx=10, pady=5)
        self.number_of_prime_tokens = tk.IntVar(value=512)
        ttk.Scale(self.root, variable=self.number_of_prime_tokens, from_=16, to=512, length=200,
                  orient=tk.HORIZONTAL).grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Select Instruments").grid(row=1, column=0, padx=10, pady=5)
        self.instruments = {'Piano': tk.BooleanVar(value=True), 'Guitar': tk.BooleanVar(value=True),
                            'Bass': tk.BooleanVar(value=False), 'Violin': tk.BooleanVar(value=True),
                            'Cello': tk.BooleanVar(value=False), 'Harp': tk.BooleanVar(value=False),
                            'Trumpet': tk.BooleanVar(value=False), 'Clarinet': tk.BooleanVar(value=False),
                            'Flute': tk.BooleanVar(value=False), 'Drums': tk.BooleanVar(value=True),
                            'Choir': tk.BooleanVar(value=False)}

        for i, (instrument, var) in enumerate(self.instruments.items(), start=2):
            ttk.Checkbutton(self.root, text=instrument, variable=var).grid(row=i, column=0, padx=10, pady=5)

        ttk.Button(self.root, text="Generate Music", command=self.generate_music).grid(row=i + 1, column=0, columnspan=2,
                                                                                      pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    model_path = "Model\Mini_Muse_Trained_Model_88000_steps_0.6129_loss.pth"
    training_data_path = "Training-Data\Mini-Muse-Training-Data.pickle"

    music_generator = MiniMuseMusicGenerator(model_path, training_data_path)
    music_generator.run()