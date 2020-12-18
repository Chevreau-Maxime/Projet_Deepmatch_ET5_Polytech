# une bonne interface console plutot que ihm pour le moment
prompt = [
    "execute deepmatch on one fragment",
    "execute ransac on one txt file"
]

def printChoices():
    print("Select option :")
    for i in range(len(prompt)):
        print(str(i) + " - " + prompt[i])
    return



done = False
while not(done):
    printChoices()
    choice = input()





