import os

# 生成目录结构
def generate_tree(path, indent=""):
    """
    :param path:遍历路径的根目录，一般为"."
    :param indent:不需要传参

    使用示例：
    generate_tree(".")
    """
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        print(f"{indent}├── {item}")
        if os.path.isdir(full_path):
            generate_tree(full_path, indent + "│   ")

if __name__ == "__main__":
    generate_tree(".")