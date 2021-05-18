#!/usr/bin/env python3


import time
import sys
import random

def format_txt(text):
    #cut long lines
    text = cut_long_lines(text)

    #indent
    text = "\n\n\t\t"+text.replace("\n","\n\t\t")
    text = find_last_non_empty_line_and_cut_cruft(text)

    return text


max_line_length=28
def cut_long_lines(text):
    lines = text.split("\n")
    tmp=""
    for l in lines:
        if len(l.rstrip())>max_line_length:
            l=insert_newlines(l.rstrip(),max_line_length)
        tmp=tmp+l+"\n"
    return tmp


def insert_newlines(string, every=40):
    words=string.split(" ")
    tmp=""
    i=0
    for word in words:
        if i>max_line_length:
            tmp+="\n"+word
            i=0
        else:
            tmp+=" "+word
        i+=len(word)
    return tmp
    #return '\n'.join(string[i:i+every] for i in range(0, len(string), every))



from string import printable
import re

import nltk.data
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

from nltk import pos_tag, word_tokenize


## EXTRACTION tools for making it all end well
forbidden=["CC","PRP","DT", "IN", "TO", "PRP$"]
poem_without_last_line=""
ll=""



# rebalance the color and genders
genders=["he","he","he","he","he","he","she","she","she","she","she","she","it","it","it","ze"]
gender_poss=["his","his","her","its"]
gender_sub=["man","man","woman","boy","girl","child"]
genderb_sub=["boy","girl","child"]
colors=["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "gray","black","white"]
polite=["African", "African-American", "slave", "black"]

def callback(ar):
    return random.choice(ar)



def find_last_non_empty_line_and_cut_cruft(crufted):

    #recursively find the last line that is not empty
    lines = crufted.split("\n")

    if len(lines)>1:
        if lines[-1].isspace() or not lines[-1] :
            return find_last_non_empty_line_and_cut_cruft("\n".join(lines[:-1]))
        else:
            poem_without_last_line="\n".join(lines[:-1])
            
            # recursively cut the cruft from last line that is not empty
            if len(lines[-1])>1:
                ll=recurse_cut_last_word_if_prp(lines[-1])

                poem=(poem_without_last_line+"\n"+ll)
                return(poem.rstrip(","))
            else:
                return (crufted.rstrip(","))
    else:
        return (crufted.rstrip(","))


def recurse_cut_last_word_if_prp (crufted):

    if len(crufted)>0:
        prelim_cr_ar=crufted.rstrip().split(" ")
       
        if len(prelim_cr_ar)>1:

            cr_ar=[None]*2
            cr_ar[0]=" ".join(prelim_cr_ar[:-1])
            cr_ar[1]=prelim_cr_ar[-1]

            wout_lastword = cr_ar[0]
            lastword=cr_ar[1].rstrip().lower()

            if lastword:
                if nltk.pos_tag([lastword])[0][1] in forbidden:
                    return recurse_cut_last_word_if_prp (wout_lastword)
                else:
                    #print("EXITTTING recurse", crufted)
                    return str(crufted)
            else:
                return (crufted)
        else:
            #SINGLE WORD LINE
            if crufted.strip() in forbidden or crufted.strip().lower() is "the" :
                #print("\nbecause the is not a line : DELETED\n", crufted)
                return ("")
            else:
                #print("single word last line KEPT", crufted)
                return str(crufted)
    else:
        #print("=== QUICK EXIT:",crufted)
        return str(crufted)



def cleanup(text): 
    # CLEANUP
    words=text

    words = words.replace(" \n","\n")        
    words = words.replace("\r","\n")
    words = words.replace("\n\n\n\n\n","\n")
    words = words.replace("\n\n\n\n","\n")
    words = words.replace("\n\n\n","\n")

    words = words.replace("\"","")
    words = words.replace("\“","")
    words = words.replace("\”","")
    
    words = words.replace(" \'","")
    words = words.replace("\' ","")
    
    words = words.replace("the the ","the ")
    words = words.replace("the  the ","the ")
    words = words.replace("The  the ","the ")
    words = words.replace("The the ","the ")
    words = words.replace(" a the ","the ")

    words = words.replace("<|endoftext|>"," ")
    

    words = words.replace("\’ ","")
    words = words.replace("\’","")

    words = words.replace(")","")
    words = words.replace("(","")
    words = words.replace("==="," ")
    words = words.replace("~ + + ~","")        

    words = words.replace("him-whose-penis-stretches-down-to-his-knees"," ")  
    words = words.replace(".the.cylinder-section.now.the.prism.cut.off.by.the."," ")

    words = words.replace("jhave@jhave-ubuntu:~/documents/github/pytorch-poetry-generation/word_language_model$"," ")
    words = words.replace("Jhave@jhave-ubuntu:~/documents/github/pytorch-poetry-generation/word_language_model$"," ")
    words = words.replace("/home/jhave/documents/github/pytorch-poetry-generation/word_language_model/data.py"," ")
    words = words.replace("/home/jhave/anaconda3/lib/python3.6/site-packages/torch/serialization.py"," ")
    words = words.replace("/home/jhave/documents/github/pytorch-poetry-generation/word_language_model'jhave@jhave-ubuntu:~/documents/github/pytorch-poetry-generation/word_language_model$"," ")
    words = words.replace("--checkpoint='/home/jhave/documents/github/pytorch-poetry-generation/word_language_model/models/2017-10-29t21-27-01/model-lstm-emsize-1600-nhid_1600-nlayers_2-batch_size_20-epoch_10-loss_6.00-ppl_402.21.pt--cuda"," ")

    # get rid of contentious distracting terminology
    words = words.replace("nigger"," ")  
    words = words.replace("niggard","cheat") 
    words = re.sub(r'\bnigger\b', callback(polite), words)
    words = re.sub(r'\bNigger\b', callback(polite), words)
    words = re.sub(r'\bNIGGER\b', callback(polite), words)
    words = re.sub(r'\bnegro\b', callback(polite), words)
    words = re.sub(r'\bNegro\b', callback(polite), words)
    words = re.sub(r'\bJew\b', callback(polite), words)
    words = re.sub(r'\bjew\b', callback(polite), words)

    words = words.replace("HoneyBrown.ca"," ")

    words = re.sub(r'\bhe\b', callback(genders), words)

    words = re.sub(r'\bhis\b', callback(gender_poss), words)

    words = re.sub(r'\bman\b', callback(gender_sub), words)
    words = re.sub(r'\bboy\b', callback(genderb_sub), words)

    words = re.sub(r'\bwhite\b', callback(colors), words)
    words = re.sub(r'\bblack\b', callback(colors), words)

    words = re.sub("[^{}]+".format(printable), "", words)

    return words






# SAVE to file
from datetime import datetime
started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
outf=open("GENERATED/"+started_datestring+".txt", "a+")
outf.write(started_datestring+"\n\nGPT-2 345M model \nretrained on custom poetry corpus \nfor ReadingRites (2019)\n\nMore info: glia.ca/rerites\n\n")

####################################
##########  ORIGINAL ###############
####################################

import fire
import json
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import model, sample, encoder

def sample_model(
    model_name='run2_April14_2020',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=80,
    temperature=1,
    min_temperature=0.3,
    max_temperature=1.3,
    pause=3,
    top_k=0,
):
    """
    Run the sample_model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """



    #SLEEP 
    time.sleep(pause)


    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:

        # Random
        temperature = random.uniform(min_temperature, max_temperature)
        #print("\n\n*********\n\nresetting temperature: ", temperature,"\n\n***********\n\n")


        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                #print(text)

                text = cleanup(text)
                #text = "\n\n\t\t\t\t"+str(round(temperature,2))+"\n\n\n"+text
                text = format_txt(text)+"\n\n"

                #display
                for char in text:
                    time.sleep(0.001)
                    sys.stdout.write(char)


                #save file
                outf.write(text)




while(True):

    if __name__ == '__main__':
        fire.Fire(sample_model)

