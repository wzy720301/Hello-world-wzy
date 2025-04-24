def print_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            print("文件内容：")
            print(''.join(lines[:9]))  # 提取前9行内容
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")

if __name__ == "__main__":
    target_file = r'e:/01-Trae-Dev/test/temp.py'
    print_file_content(target_file)
