"""演示函数定义与调用。"""


def greet(name: str, prefix: str = "Hello") -> str:
    return f"{prefix}, {name}!"


def max_of_three(a: float, b: float, c: float) -> float:
    return max(a, b, c)


def average(nums: list[float]) -> float:
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


if __name__ == "__main__":
    print(greet("Alice"))
    print(greet("Bob", prefix="Hi"))
    print("max_of_three:", max_of_three(3, 7, 5))
    print("average:", average([1, 2, 3, 4, 5]))

