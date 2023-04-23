#GAN-Based Bandwidth Extension for Music
This is a Generative Adversarial Network (GAN) that creates the missing high-frequency content
for low-bandwidth music. It uses the WaveNet architecture and multi-domain and multi-scale
discriminators.

This was the final project for Computer Audition (ECE477).


# Usage
To train the model, use [training.py](training.py). To run the existing model on a single
audio file, use [run.py](run.py). To evaluate objective results on the entire dataset, use 
[eval.py](eval.py).

Parameters are stored in [config.yml](config.yml).
