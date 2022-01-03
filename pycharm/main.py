import pandas as pd
import numpy as np
import matplotlib as plt
import os

def palindrome(word):
    print("단어는 : ", word)
    if len(word) < 2:
        return True
    if word[0] != word[-1]:
        print(word[0],"와(과) ",word[-1],"은(는) 같지 않다.")
        return False
    return palindrome(word[1:-1])

print(palindrome('hello'))
print(palindrome('토마토마토마토'))