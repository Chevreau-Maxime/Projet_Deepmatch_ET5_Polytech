import os

# une bonne interface console plutot que ihm pour le moment
prompt = [
    "execute deepmatch on one fragment",
    "execute ransac on one txt file",
    "initialize images"
]

def printChoices():
    print("Select option :")
    for i in range(len(prompt)):
        print(str(i) + " - " + prompt[i])
    return

def Main():
    done = False
    while not(done):
        os.system('CLS')
        printChoices()
        option = input()
        frag = input("Choose a fragment number : ")
        if (option==0):
            return
        if (option==1):
            os.system('python __ransac.py resultats3/'+str(frag)+'.txt')
        if (option==2):
            os.system('python _init_fresque.py')


Main()













