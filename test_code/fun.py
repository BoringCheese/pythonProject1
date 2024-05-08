i = 1


def ttt():
    global i
    print(i)
    i = i + 1


if __name__ == '__main__':
    for a in range(3):
        ttt()
