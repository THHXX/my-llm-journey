"""演示条件判断与循环。"""

score = 85

if score >= 90:
    grade = "A"
elif score >= 75:
    grade = "B"
else:
    grade = "C"

print(f"成绩 {score} -> 等级 {grade}")

# for 遍历列表
nums = [1, 2, 3, 4, 5]
sum_val = 0
for n in nums:
    sum_val += n
print("sum via for:", sum_val)

# while 计数器
count = 0
while count < 3:
    print("while loop count:", count)
    count += 1

# range 使用
for i in range(2, 10, 3):  # start=2, stop=10(不含), step=3
    print("range item:", i)


