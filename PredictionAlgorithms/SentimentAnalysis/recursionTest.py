class RecursionTest():
    def __init__(self):
        pass

    def power(self, xBase, yPower):
        if yPower == 0:
            return 1
        else:
            print(xBase * self.power(xBase, yPower - 1))
            return xBase * self.power(xBase, yPower - 1)

if __name__ == ("__main__"):
    recur = RecursionTest().power(2,3)
    print(recur)

