"""演示变量与基本数据类型。"""

# 数值与字符串
integer_val = 42
float_val = 3.14
string_val = "hello"
bool_val = True

print("整数:", integer_val, type(integer_val))
print("浮点数:", float_val, type(float_val))
print("字符串:", string_val, type(string_val))
print("布尔值:", bool_val, type(bool_val))

# 序列与集合类型
list_val = [1, 2, 3]
tuple_val = (1, 2, 3)
set_val = {1, 2, 3}
dict_val = {"name": "Alice", "age": 20}

print("列表:", list_val, type(list_val))
print("元组:", tuple_val, type(tuple_val))
print("集合:", set_val, type(set_val))
print("字典:", dict_val, type(dict_val))

# 常用操作
list_val.append(4)
dict_val["city"] = "Beijing"

print("列表长度:", len(list_val))
print("切片示例:", list_val[1:3])
print("字典 keys:", list(dict_val.keys()))


