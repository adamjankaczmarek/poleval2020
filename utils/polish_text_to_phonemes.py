#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import pathlib
import sys

Vowels = set("aiouey") | set(["on", "en"])

Dzwieczne = {
    "w": "f", "d": "t", "drz": "cz", "rz": "sz", "dzi": "ci", "dz": "c", "b": "p", "z": "s", "g": "k", "zi": "si",
    'gi': 'ki'
}

Bezdzwieczne = {
    "f": "w", "t": "d", "cz": "drz", "sz": "rz", "ci": "dzi", "c": "dz", "p": "b", "s": "z", "k": "g", "si": "zi",
    "ch": "ch", 'ki': 'gi'
}

Stops = set("bpgkdt")

BASE_DIR = pathlib.Path(__file__).parent.absolute()


def readRules(n):
    rv = []
    for x in open(n,"r"):
      if x[0] == "#": continue  # komentarz
      L = x.split()
      if len(L) != 2: continue
      rv.append(L)
    return rv
        
def applyRules(R,s):
   for a,b in R:
     s = s.replace(a,b)
   #print "AR",s  
   return s

def miekkie(L):
  #print "miekkie",L
  rv = []
  i = 0
  while i < len(L):
    if i < len(L)-2 and L[i+1] == "i" and L[i+2] in Vowels:
       if L[i][-1] == "i":
         rv.append(L[i])
         rv.append(L[i+2])
         i += 3
       else:
         rv.append(L[i])
         rv.append("j")   #tymczasowo usuniemy
         rv.append(L[i+2])
         i += 3
    else:
      rv.append(L[i])
      i += 1
  return rv
        
def swap(g,D):
  if D == "D":
    if g in Bezdzwieczne:
       return Bezdzwieczne[g]
  else:
    if g in Dzwieczne:
       return Dzwieczne[g]
  return g    

R1 = readRules(os.path.join(BASE_DIR, "data/reguly.txt"))
LG = readRules(os.path.join(BASE_DIR, "data/litery_gloski.txt"))

def wsteczne(L):
  L = L[:]
  i = len(L) - 1
  Last = "?"
  while i>=0:
    if Last == "?":
      if L[i] in Dzwieczne:
        Last= "D"
      elif L[i] in Bezdzwieczne:
        Last = "B"
    if L[i] in Vowels:
       Last = "?"
    if Last != "?":
       L[i] = swap(L[i],Last)
    i -= 1
  return L  
       
def say_word(s):
   if s == '<unk>':
       return '#' 
   s = "_" + s+ "_"
   s = applyRules(R1,s)
   if s[-1] == "_":
      s = s[:-1]
   if s[0] == "_":
      s = s[1:]   
   s = "-".join(s)

   # print( "After rules",s)

   G0 = applyRules(LG,"-"+s+"-")
   G0 = G0[1:-1]
   
   p = G0.find("n-t-sz") 
   
   G0 = G0.replace("en","e-ng")
   G0 = G0.replace("on","o-ng")
   G0 = G0.replace("n-k","ng-k")
   G0 = G0.replace("n-g","ng-g")
   G0 = G0.replace("ch","h")  # nowe oznaczenie
   
   
   if p != -1:
     p_s = p + 7
     if p_s < len(G0):
        if not G0[p_s:p_s+1] in Vowels:
           G0 = G0.replace("n-t-sz","n-cz")
   

   G = miekkie(G0.split("-"))
   #print (G, G0)     
   return new_names(wsteczne(G))

new_mapping = {
   'sz' : 'š',
   'dzi' : 'ď',
   'cz' : 'č',
   'ni' : 'ń',
   'si' : 'ś',
   'ci' : 'ć',
   'zi' : 'ź',
   'rz' : 'ż',
   'll' : 'ł',
   'drz' : 'ž',
   'dz' : 'ʒ',
   'ng' : 'ŋ',
   'ki' : 'kj',
   'gi' : 'gj'
}

def new_names(s):
    res = []
    for p in s:
        if p in new_mapping:
            res.append(new_mapping[p])
        else:
            res.append(p)
    return ''.join(res).replace('ji', 'i')

def wz_glue(L):
    c = ''
    res = []
    for w in L:
        if w == 'w' or w == 'z':
            c = w
        else:
            if w != '<unk>':
                res.append(c+w)
            else:
                res += [c, w]    
            c = ''
    if c:
        res.append(c)
    return res
    
def say_sentence(L):
    return ''.join([say_word(w) for w in wz_glue(L)])

  
if __name__ == "__main__":
   for x in sys.stdin:
      x = x.strip()
      print (say_sentence(wz_glue(x.split())))
          

