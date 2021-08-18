

def find(word):
    result = False
    print("111111111111")
    while True:
        print("222222222222222222{}".format(word))
        line = (yield result)
        print("3333333333333333333333{}-{}".format(word, line))
        result = word in line
        print("444444444444444444{}-{}".format(word, line))



f = find("Python")
next(f)
print("---------------------------------------------------------------")
print(f.send("Hellow, Python!"))
print("---------------------------------------------------------------")
print(f.send("Hellow, world!"))
print("---------------------------------------------------------------")
print(f.send("Python Script"))
print("---------------------------------------------------------------")

f.close()