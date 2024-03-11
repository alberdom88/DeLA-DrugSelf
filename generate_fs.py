import torch
import preProcess
from preProcess import chemblData
import torch.nn as nn
import random
import sys
import rdkit.Chem as Chem
import rdkit.Chem.rdchem
import SAScore
import os
import enchant
import datasets


from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import selfies
selfies.bond_constraints.set_semantic_constraints({'H': 1, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 3, 'B': 3, 'B+1': 2, 'B-1': 4, 'O': 2, 'O+1': 3, 'O-1': 1, 'N': 3, 'N+1': 4, 'N-1': 2, 'C': 4, 'C+1': 5, 'C-1': 3, 'P': 5, 'P+1': 6, 'P-1': 4, 'S': 6, 'S+1': 7, 'S-1': 5, '?': 8})

vocSelfie = {'A':'[C]', 'B':'[11C]', 'C':'[13CH1]', 'D':'[3H]', 'E':'[O]', 'F':'[C-1]', 'G':'[17F]', 'H':'[Branch2]', 'I':'[=P]', 'J':'[#11C]', 'K':'[11CH3]', 'L':'[=S+1]', 'M':'[11CH2]', 'N':'[131I]', 'O':'[=N]', 'P':'[N]', 'Q':'[=Branch2]', 'R':'[#Branch2]', 'S':'[Br]', 'T':'[#C-1]', 'U':'[#N]', 'V':'[123I]', 'W':'[#N+1]', 'X':'[=13CH1]', 'Y':'[F]', 'Z':'[#C]', 'a':'[C+1]', 'b':'[14CH1]', 'c':'[I+1]', 'd':'[=S]', 'e':'[=N+1]', 'f':'[14CH2]', 'g':'[#14C]', 'h':'[N-1]', 'i':'[#Branch1]', 'j':'[O-1]', 'k':'[125I]', 'l':'[15N]', 'm':'[I]', 'n':'[P]', 'o':'[18F]', 'p':'[=Ring1]', 'q':'[11CH1]', 'r':'[=C]', 's':'[P+1]', 't':'[75Br]', 'u':'[18OH1]', 'v':'[S+1]', 'w':'[Cl]', 'x':'[=N-1]', 'y':'[SH0]', 'z':'[=O+1]', '0':'[CH1-1]', '1':'[=SH0]', '2':'[=14CH1]', '3':'[=Ring2]', '4':'[35S]', '5':'[2H]', '6':'[14C]', '7':'[=18O]', '8':'[Branch1]', '9':'[14CH3]', '<':'[N+1]', '>':'[=Branch1]', '&':'[=O]', '!':'[O+1]', '=':'[32P]', '+':'[=14C]', '-':'[Ring2]', '[':'[13C]', ']':'[=11C]', '{':'[Ring1]', '}':'[S]', '/':'[NH1]', '_':'[124I]', '£':'[15NH1]'}

#vocSelfie = {'A':'[C]', 'E':'[O]', 'F':'[C-1]', 'H':'[Branch2]', 'I':'[=P]', 'L':'[=S+1]', 'O':'[=N]', 'P':'[N]', 'Q':'[=Branch2]', 'R':'[#Branch2]', 'S':'[Br]', 'T':'[#C-1]', 'U':'[#N]', 'W':'[#N+1]', 'Y':'[F]', 'Z':'[#C]', 'a':'[C+1]', 'c':'[I+1]', 'd':'[=S]', 'e':'[=N+1]', 'h':'[N-1]', 'i':'[#Branch1]', 'j':'[O-1]', 'm':'[I]', 'n':'[P]', 'p':'[=Ring1]', 'r':'[=C]', 's':'[P+1]', 'v':'[S+1]', 'w':'[Cl]', 'x':'[=N-1]', 'y':'[SH0]', 'z':'[=O+1]', '0':'[CH1-1]', '1':'[=SH0]', '3':'[=Ring2]', '8':'[Branch1]', '<':'[N+1]', '>':'[=Branch1]', '&':'[=O]', '!':'[O+1]', '-':'[Ring2]', '{':'[Ring1]', '}':'[S]', '/':'[NH1]'}        

def sampleDistribution(dist):
    ran = random.random()
    s = 0
    for i in range(0, len(dist)):
        s = s + dist[i]
        if s > ran:
            return i 

class generatorFS():
    def __init__(self, paramsDict):
        self.device = paramsDict["device"]
        self.net = torch.load(paramsDict["networkPath"], map_location=torch.device('cpu')).to(self.device)
        fh = open(paramsDict["vocabularyPath"])
        self.voc = eval(fh.read())
        fh.close()
        self.contextLength= paramsDict["contextLength"]
        self.vocLength = len(self.voc)
        self.softmax = nn.Softmax(dim = -1)
        self.wantedValids = 1

    def run(self):
        valid = 0
        while valid < self.wantedValids:
            over = False
            answer = datasets.oneHotEncode("<s>", self.voc).unsqueeze(0)
            hx1s = torch.zeros(1, self.net.hiddenSize_L1)
            cx1s = torch.zeros(1, self.net.hiddenSize_L1)
            hx2s = torch.zeros(1, self.net.hiddenSize_L2)
            cx2s = torch.zeros(1, self.net.hiddenSize_L2)
            contexts  = torch.zeros(1, self.net.inputSize)
            while over is False:
                contexts = datasets.addToContext(contexts, answer[-1].unsqueeze(0))
                output, hx1s, cx1s, hx2s, cx2s = self.net(contexts, hx1s, cx1s, hx2s, cx2s)
                distribution = self.softmax(output[0])
                sampledIndex = sampleDistribution(distribution)
                nextChar = torch.zeros(1,len(self.voc))
                nextChar[0, sampledIndex] = 1
                answer = torch.cat((answer, nextChar), dim = 0)
                
                if datasets.oneHotDecode(nextChar, self.voc)== "</s>" or datasets.oneHotDecode(nextChar, self.voc)== "~":
                    over = True 

            selfie = datasets.oneHotDecodeWord(answer[1:-1], self.voc)
            #print(selfie)
            try:
                if (len(selfie)==0):
                    continue
                m1= ""
                for l in selfie:
                    m1+=vocSelfie[l]
                if (len(m1)==0):
                    continue
                m2 = selfies.encoder(selfies.decoder(m1))
                a1 = selfie
                a2 = m2
                for j in vocSelfie.values():
                    a2 = a2.replace(j,list(vocSelfie.keys())[list(vocSelfie.values()).index(j)])
                a2 = a2.replace("~","")    
                if (a2!=a1):
                    dist = enchant.utils.levenshtein(a2,a1)+1.0#/(max(len(a1),len(a2))*(1.0)) + 1
                else:
                    dist = 1.0
                #print(dist)
                valid = valid + 1
            except:
                continue     
        
        return selfie


scriptPath = os.path.dirname(os.path.realpath(__file__)) 

params={"vocabularyPath": scriptPath + "/models/training_full_dataset_c12.voc",
         "networkPath": scriptPath + "/models/training_full_dataset_c12.net",
         "contextLength": 12,
         "device" : "cpu",
         "out": sys.argv[1], 
         "nmols": sys.argv[2]
         }

f = open(params["out"],"w")
gen = generatorFS(params)
for i in range(int(params["nmols"])):
    print ("Gen: "+str(i+1))
    m = gen.run()
    answer = ""
    for c in m:
        answer+=vocSelfie[c]
    f.write(selfies.decoder(answer)+"\n")

f.close()

