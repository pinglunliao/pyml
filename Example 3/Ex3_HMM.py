# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:13:52 2018

@author: YunAIUser
"""

from pomegranate import HiddenMarkovModel, DiscreteDistribution, State
import random
import numpy as np
from tkinter import Tk, Label, Entry, Button, messagebox, LEFT

seqIn = "abcdaaaacaddbcd"
random.seed(0)
hmmModel = HiddenMarkovModel( "HW3 HMM" )   

def modelInit():
    # Initial probability for three states
    state_one  = State( DiscreteDistribution({'a': 0.1, 'b':0.2, 'c': 0.2, 'd':0.5 }), name="S1" )
    state_two = State( DiscreteDistribution({'a': 0.2, 'b':0.5, 'c': 0.15, 'd':0.15 }), name="S2" )
    state_three = State( DiscreteDistribution({'a': 0.2, 'b':0.15, 'c': 0.15, 'd':0.5 }), name="S3" )
    
    # starting transition
    hmmModel.add_transition( hmmModel.start, state_one, 0.1 )
    hmmModel.add_transition( hmmModel.start, state_two, 0.1 )
    hmmModel.add_transition( hmmModel.start, state_three, 0.8 )
    
    # Transition matrix
    hmmModel.add_transition( state_one, state_one, 0.4 )
    hmmModel.add_transition( state_one, state_two, 0.3 )
    hmmModel.add_transition( state_one, state_three, 0.3 )
    hmmModel.add_transition( state_two, state_one, 0.3 )
    hmmModel.add_transition( state_two, state_two, 0.4 )
    hmmModel.add_transition( state_two, state_three, 0.3 )
    hmmModel.add_transition( state_three, state_one, 0.3 )
    hmmModel.add_transition( state_three, state_two, 0.3 )
    hmmModel.add_transition( state_three, state_three, 0.4 )
    
    hmmModel.bake( verbose=True )

def readTextData(fName):
    fIn = open(fName, "r", encoding = "utf-8")
    lines = fIn.read().split()
    
    sequences = []
    
    for line in lines:
        sequences.append( np.array(list(line)) )
    return sequences

def hmmPredict():
    seqIn = seqTxt.get()
    seqIn = np.array(list(seqIn))
    rTxt = ""
    try:
        logp, path = hmmModel.viterbi( seqIn )
        rTxt = "序列:'{}' --Log 機率:{} --路徑:{}".format(''.join( seqIn ),
                logp, " ".join( state.name for idx, state in path[1:-1] ) ) 
        print(rTxt)        
    except:
        rTxt = "請輸入只含有(abcd)的序列"

    resultLabel.config(text = rTxt)
    
modelInit()

print( hmmModel.dense_transition_matrix()[hmmModel.start_index, 0:3] )
print( hmmModel.dense_transition_matrix()[1:-1, 0:3] )
seqIn = np.array(list(seqIn))
logp, path = hmmModel.viterbi( seqIn )

trainSeq = readTextData("hmmData.txt")
hmmModel.fit(trainSeq)

print("After training")
print( hmmModel.dense_transition_matrix()[hmmModel.start_index, 0:3] )
print( hmmModel.dense_transition_matrix()[1:-1, 0:3] )


mainWin = Tk()

# 視窗標題
mainWin.title("機器學習作業3:隱藏式馬柯夫模型(HMM)")

# 視窗大小
mainWin.geometry("800x50")

# 建立GUI元件
seqLabel = Label( mainWin, text = "輸入序列:" )
seqTxt = Entry( mainWin )
seqTxt.focus()
hmmPredictBtn = Button( mainWin, text = "預測", command = hmmPredict )
resultLabel = Label( mainWin, text = "結果：" )

# GUI Layout
seqLabel.pack(padx=5, pady=1, side=LEFT) #grid(row=0, column=0)
seqTxt.pack(padx=5, pady=1, side=LEFT) #grid(row=0, column=1)
hmmPredictBtn.pack(padx=5, pady=1, side=LEFT) #grid(row=0, column=2)
resultLabel.pack(side=LEFT) #grid(row=1, column=1)

mainWin.mainloop()
