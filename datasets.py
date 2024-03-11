import random
import torch
import selfies


def addToContext(context, letter):
    newContext = torch.cat((context, letter), dim = 1)[:, letter.shape[1]:]
    return newContext

def oneHotDecode(encodedCharacter, vocabulary):
    out = vocabulary[int(torch.argmax(encodedCharacter).item())]
    return out


def oneHotDecodeWord(encodedWord, vocabulary):
    outWord = ""
    for i in encodedWord:
        outWord += oneHotDecode(i, vocabulary)
    return outWord
        

def oneHotEncode(character, vocabulary):
    out = torch.zeros(len(vocabulary))
    out[vocabulary.index(character)] = 1
    return out


def oneHotEncodeWord(word, voc):
    out = torch.zeros(len(word), len(voc))
    for c in range(0, len(word)):
        out[c] = oneHotEncode(word[c], voc)
    return out

