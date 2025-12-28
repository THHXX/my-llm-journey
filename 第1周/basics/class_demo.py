"""演示类与对象。"""


class Student:
    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self.score = score

    def is_pass(self) -> bool:
        return self.score >= 60

    def describe(self) -> str:
        status = "及格" if self.is_pass() else "不及格"
        return f"{self.name} 成绩 {self.score} -> {status}"


if __name__ == "__main__":
    alice = Student("Alice", 85)
    bob = Student("Bob", 55)
    print(alice.describe())
    print(bob.describe())

