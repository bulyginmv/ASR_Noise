# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 04:20:04 2020

@author: Misha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:46:56 2020
@author: sleek_eagle
"""
import os
import math   
import numpy as np 
import soundfile as sf                                                      
import matplotlib.pyplot as plt
import librosa
import speech_recognition as sr
import wavio as w
import pandas as pd
import random
from nltk.translate.bleu_score import sentence_bleu


r = sr.Recognizer()
#load transcription
lowest_dirs = list()
for root,dirs,files in os.walk('D:/hello/ASR/LibriSpeech/test-clean/'):
    if not dirs:
        lowest_dirs.append(root)

def read_transcription(path_to_trans):
    with open(path_to_trans) as trans:
        trans = trans.read().split('\n')
        for i,t in enumerate(trans):
            index = t.find(' ')
            trans[i] = t[index+1:].lower()
        return trans

def make_prediction (audio_path):

    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        pred = r.recognize_ibm(audio, username = '3059e1fd-bac3-42f0-8d3e-ac826a71842c', password = '71c5bca4f41043a187c1b79f91fc2cf7')

        return pred
    except sr.UnknownValueError:
        pred = None
        return pred
    except sr.RequestError as e:
        pred = None
        return pred


def calculate_error(truth,pred):
    if pred == None:
        wer = wil = mer = wip = 1
        return wer, wil, mer, wip
    wer = j.wer(truth, pred)
    wil = j.wil(truth,pred)
    mer = j.mer(truth,pred)
    bleu = sentence_bleu([truth.split()], pred.split(),weights = (0.7,0.25,0.04,0.01))
    bleu = abs(bleu - 1)
    return wer, wil, mer, bleu

def make_color_noise(beta, audio_path):
    signal, freq = sf.read(audio_path)
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    samples = len(signal)
    noise = cn.powerlaw_psd_gaussian(beta, samples)
    noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))
    return noise



'''
Signal to noise ratio (SNR) can be defined as 
SNR = 20*log(RMS_signal/RMS_noise)
where RMS_signal is the RMS value of signal and RMS_noise is that of noise.
      log is the logarithm of 10
*****additive white gausian noise (AWGN)****
 - This kind of noise can be added (arithmatic element-wise addition) to the signal
 - mean value is zero (randomly sampled from a gausian distribution with mean value of zero. standard daviation can varry)
 - contains all the frequency components in an equal manner (hence "white" noise) 
'''

#SNR in dB
#given a signal and desired SNR, this gives the required AWGN what should be added to the signal to get the desired SNR
def get_white_noise(signal,SNR) :
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/20)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    return noise

def add_color_noise(signal,freq, noise,SNR):
    RMS_s = math.sqrt(np.mean(signal ** 2))
    # required RMS of noise
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, SNR / 20)))

    # current RMS of noise
    RMS_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (RMS_n / RMS_n_current)

    noise_signal = signal + noise
    path = 'temp.wav'
    w.write("temp.wav", noise_signal, freq, sampwidth=3)
    return path


#given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
def get_noise_from_sound(signal, freq, noise_file,SNR):
    noise, fr = librosa.load(noise_file)
    noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))

    if (len(noise)<len(signal)):
        while (len(noise)<len(signal)):
            noise_temp = noise.copy()
            noise = np.concatenate((noise,noise_temp))

    if(len(noise)>len(signal)):
        noise=noise[0:len(signal)]

    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/20)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    noise_signal = signal + noise
    path = 'temp.wav'
    w.write("temp.wav", noise_signal, freq, sampwidth=3)
    return path

#***convert complex np array to polar arrays (2 apprays; abs and angle)
def to_polar(complex_ar):
    return np.abs(complex_ar),np.angle(complex_ar)


def add_noise (noise_path,signal, freq, truth, SNR):
    noisy_path = get_noise_from_sound(signal,freq, noise_path, SNR)

    prediction_google = make_prediction(noisy_path)

    wer, wil, mer, bleu = calculate_error(truth, prediction_google)
    return wer, wil, mer, bleu


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def demon_noise(signal,chunk,freq):
    noise = list(chunks(signal, chunk))
    noise = random.sample(noise, len(noise))
    noise = np.concatenate(noise,axis = 0)
    path  = 'noise.wav'
    w.write("noise.wav", noise, freq, sampwidth=3)
    return path


#**********************************
#*************add AWGN noise******
#**********************************

df = pd.DataFrame(columns = ['Audio','Clean_WER','Clean_WIL','Clean_MER','Clean_BLEU',
                             'whitenoise_WER','whitenoise_WIL','whitenoise_MER','whitenoise_BLEU',
                             'pinknoise_WER','pinknoise_WIL','pinknoise_MER','pinknoise_BLEU',
                             'brownnoise_WER','brownnoise_WIL','brownnoise_MER','brownnoise_BLEU',
                             'violetnoise_WER','violetnoise_WIL','violetnoise_MER','violetnoise_BLEU',
                             'bluenoise_WER','bluenoise_WIL','bluenoise_MER','bluenoise_BLEU',
                             'dog_bark_WER','dog_bark_WIL','dog_bark_MER','dog_bark_BLEU',
                             'streets_WER','streets_WIL','streets_MER','streets_BLEU',
                             'talking_WER','talking_WIL','talking_MER','talking_BLEU',
                             'roadworks_WER', 'roadworks_WIL', 'roadworks_MER', 'roadworks_BLEU',
                             'mixtalk_WER', 'mixtalk_WIL', 'mixtalk_MER', 'mixtalk_BLEU'])

noises = ['pinknoise.wav','brownnoise.wav','violetnoise.wav','bluenoise.wav','dog_bark.wav','streets.wav','talking.wav','roadworks.wav']

count = 0

for dir in lowest_dirs:

    l = os.listdir(dir)
    path_to_trans = os.path.join(dir, l[-1])
    trans = read_transcription(path_to_trans)

    l = l[:-1]

    for index, file in enumerate(l):

        df_temp = pd.DataFrame(index=range(1),columns = ['Audio','Clean_WER','Clean_WIL','Clean_MER','Clean_BLEU',
                             'whitenoise_WER','whitenoise_WIL','whitenoise_MER','whitenoise_BLEU',
                             'pinknoise_WER','pinknoise_WIL','pinknoise_MER','pinknoise_BLEU',
                             'brownnoise_WER','brownnoise_WIL','brownnoise_MER','brownnoise_BLEU',
                             'violetnoise_WER','violetnoise_WIL','violetnoise_MER','violetnoise_BLEU',
                             'bluenoise_WER','bluenoise_WIL','bluenoise_MER','bluenoise_BLEU',
                             'dog_bark_WER','dog_bark_WIL','dog_bark_MER','dog_bark_BLEU',
                             'streets_WER','streets_WIL','streets_MER','streets_BLEU',
                             'talking_WER','talking_WIL','talking_MER','talking_BLEU',
                             'roadworks_WER', 'roadworks_WIL', 'roadworks_MER', 'roadworks_BLEU',
                             'mixtalk_WER', 'mixtalk_WIL', 'mixtalk_MER', 'mixtalk_BLEU'])

        signal_file = os.path.join(dir, file)

        df_temp['Audio'] = signal_file

        signal, freq = sf.read(signal_file)
        signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))

        prediction_google = make_prediction(signal_file)

        truth = trans[index]

        wer, wil, mer, bleu = calculate_error(truth,prediction_google)

        df_temp['Clean_WER'] = wer
        df_temp['Clean_WIL'] = wil
        df_temp['Clean_MER'] = mer
        df_temp['Clean_BLEU'] = bleu

        white_noise = get_white_noise(signal,SNR=10)

        noisy_path = add_color_noise(signal,freq,white_noise,10)

        prediction_google_white = make_prediction(noisy_path)

        wer, wil, mer, bleu = calculate_error(truth,prediction_google_white)

        df_temp['whitenoise_WER'] = wer
        df_temp['whitenoise_WIL'] = wil
        df_temp['whitenoise_MER'] = mer
        df_temp['whitenoise_BLEU'] = bleu



        for noise in noises:
            wer, wil, mer, bleu = add_noise(noise,signal,freq,truth,SNR = 10)
            df_temp[noise[:-4]+'_WER'] = wer
            df_temp[noise[:-4]+'_WIL'] = wil
            df_temp[noise[:-4]+'_MER'] = mer
            df_temp[noise[:-4]+'_BLEU'] = bleu



        noise_path = demon_noise(signal,1000,freq)
        wer, wil, mer, bleu = add_noise(noise_path, signal,freq,truth, SNR = 10)
        df_temp['mixtalk_WER'] = wer
        df_temp['mixtalk_WIL'] = wil
        df_temp['mixtalk_MER'] = mer
        df_temp['mixtalk_BLEU'] = bleu

        df = df.append(df_temp)
        print('Done with file', signal_file)

        count +=1
        if count == 1000:
            break
        print('Processed', str(count/10)+'%')
    if count == 1000:
        break
df.to_csv('houndify_speech_results.csv')

