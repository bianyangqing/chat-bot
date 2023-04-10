import random

def add_random_chars(string):
    result = ""
    for char in string:
        random_char = chr(random.randint(33, 126))  # 生成33到126之间的随机ASCII码
        result += char + random_char
    return result

# 示例用法
my_string = "xxx"
new_string = add_random_chars(my_string)
print(new_string)
result = new_string[::2]
print(result)