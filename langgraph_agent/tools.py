from langchain_core.tools import tool
import os
import subprocess


@tool
def create_file(f_name, f_content):
    """创建一个新文件，写入指定内容。
    注意：若f_name为绝对路径（如`/tmp/test.txt`），则直接使用该路径；
          若为相对路径（如`message/test.txt`），则基于当前工作目录拼接路径。
    Args:
        f_name: 文件路径（相对或绝对）。
        f_content: 要写入的文本内容。
    Returns:
        统一格式的结果字典：
            - success: 布尔值，操作是否成功
            - message: 成功时的详细信息（如文件路径）
            - error: 失败时的错误信息（空字符串表示无错误）
    """
    try:
        if os.path.isabs(f_name):
            f_pth = f_name
        else:
            f_pth = os.path.join(os.getcwd(), f_name)
        os.makedirs(os.path.dirname(f_pth), exist_ok=True)
        with open(f_pth, 'w') as f:
            f.write(f_content)
        return {
            "success": True,
            "message": f"File '{f_name}' created successfully at '{f_pth}'.",
            "error": ""
        }
    except Exception as e:
        return {
            "success": False,
            "message": "",
            "error": f"Failed to create file: {str(e)}"
        }


@tool
def str_replace(f_name, old_str, new_str):
    """在指定文件中替换第一个出现的字符串, 路径规则同create_file。
    Args:
        f_name: 文件路径（相对或绝对）。
        old_str: 要被替换的旧字符串。
        new_str: 用于替换的新字符串。
    Returns:
        统一格式的结果字典（同create_file）。
    """
    try:
        if os.path.isabs(f_name):
            f_pth = f_name
        else:
            f_pth = os.path.join(os.getcwd(), f_name)

        if not os.path.exists(f_pth):
            return {
                "success": False,
                "message": "",
                "error": f"File '{f_pth}' does not exist."
            }

        # 流式处理（避免大文件占用过多内存）
        replaced = False
        temp_pth = f"{f_pth}.tmp"  # 临时文件存储结果

        with open(f_pth, 'r') as infile, open(temp_pth, 'w') as outfile:
            for line in infile:
                if not replaced:
                    # 替换当前行中第一个出现的目标字符串
                    new_line = line.replace(old_str, new_str, 1)
                    outfile.write(new_line)
                    # 检查是否完成替换
                    if new_line != line:
                        replaced = True
                else:
                    # 替换完成后直接写入剩余内容
                    outfile.write(line)

        # 替换原始文件
        os.replace(temp_pth, f_pth)

        if replaced:
            return {
                "success": True,
                "message": f"Replaced first occurrence of '{old_str}' with '{new_str}' in {f_pth}.",
                "error": ""
            }
        else:
            return {
                "success": True,
                "message": f"No occurrence of '{old_str}' found in {f_pth}.",
                "error": ""
            }

    except Exception as e:
        # 清理临时文件（若存在）
        if os.path.exists(temp_pth):
            os.remove(temp_pth)
        return {
            "success": False,
            "message": "",
            "error": f"Failed to replace string: {str(e)}"
        }


@tool
def send_message(message: str):
    """发送一条消息给用户。
    Args:
        message: 要发送的消息内容。
    Returns:
        统一格式的结果字典。
    """
    try:
        return {
            "success": True,
            "message": message,  # 传回消息内容
            "error": ""
        }
    except Exception as e:
        return {
            "success": False,
            "message": "",
            "error": f"Failed to send message: {str(e)}"
        }


import shlex

# 定义允许执行的shell命令白名单（限制安全风险）
ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "wc",
    "python", "python3", "pip", "pip3", "echo"
}
DANGEROUS_COMMANDS = {"rm", "mv", "cp", "sudo", "chmod", "chown", "rmdir", "mkdir"}  # 明确禁止的危险命令


def _is_safe_command(command: str) -> bool:
    """检查命令是否安全（在白名单中且不包含危险命令）"""
    # 拆分命令获取主命令（忽略参数）
    try:
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return False
        main_cmd = cmd_parts[0].lower()
        # 禁止危险命令，允许白名单命令
        return main_cmd not in DANGEROUS_COMMANDS and (main_cmd in ALLOWED_COMMANDS)
    except Exception:
        return False


@tool
def shell_exec(command: str):
    """执行一个安全的shell命令并返回结果（限制危险操作）。
    仅允许执行白名单内的命令（ls、cat、python等），禁止rm、sudo等危险命令。
    Args:
        command: 要执行的shell命令字符串。
    Returns:
        统一格式的结果字典：
            - success: 布尔值（命令执行成功且返回码为0）
            - message: 包含stdout、stderr、returncode的字典
            - error: 错误信息（如命令不被允许、执行失败等）
    """
    # 检查命令安全性
    if not _is_safe_command(command):
        return {
            "success": False,
            "message": {},
            "error": f"Command '{command}' is not allowed (security restriction)."
        }

    try:
        # 拆分命令为列表（避免shell=True的安全风险）
        cmd_parts = shlex.split(command)

        # 执行命令（禁用shell模式）
        result = subprocess.run(
            cmd_parts,
            shell=False,  # 关键：禁用shell模式，防止注入
            capture_output=True,
            timeout=30  # 限制命令执行时间（避免超时）
        )

        # 解码输出（支持多编码）
        try:
            stdout = result.stdout.decode('utf-8')
            stderr = result.stderr.decode('utf-8')
        except UnicodeDecodeError:
            stdout = result.stdout.decode('gbk', errors='replace')
            stderr = result.stderr.decode('gbk', errors='replace')

        # 判定成功与否（returncode为0视为成功）
        success = result.returncode == 0

        return {
            "success": success,
            "message": {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode
            },
            "error": "" if success else f"Command exited with code {result.returncode}: {stderr}"
        }

    except Exception as e:
        return {
            "success": False,
            "message": {},
            "error": f"Failed to execute command: {str(e)}"
        }