class A:
    def __init__(self):
        self.a = "ccc"


print(A().a)
print(getattr(A(), "a"))
