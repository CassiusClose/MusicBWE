# GAN-Based Bandwidth Extension for Music
This is a Generative Adversarial Network (GAN) that creates the missing high-frequency content
for low-bandwidth music. It uses the WaveNet architecture and multi-domain and multi-scale
discriminators.

This was the final project for Computer Audition (ECE477).

## Examples
Sample results can be found in the [samples/](samples/) directory. There are both segments of
the song and the entire song. The ground truth example ends in "original.wav", and the
low-bandwidth input file ends in "lp.wav". The final result is stored in "fullmodel.wav".

There are also ablation tests where various parts of the model were removed: "nodisc.wav"
uses only the generator, and "nospect.wav" uses only time-domain discriminators (and not
frequency-domain discriminators).

## Usage
To train the model, use [training.py](training.py). To run the existing model on a single
audio file, use [run.py](run.py). To evaluate objective results on the entire dataset, use 
[eval.py](eval.py).

Parameters are stored in [config.yml](config.yml).
