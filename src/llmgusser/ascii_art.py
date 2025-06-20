import os

# ANSIカラーコード
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    
    # 前景色（文字色）
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # 背景色
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


# パターン3: アスキーアートスタイル
def display_question_ascii(question_num):
    print(f"\n{Colors.BRIGHT_MAGENTA}")
    print("  ██████╗ ██╗   ██╗███████╗███████╗████████╗██╗ ██████╗ ███╗  ██╗")
    print("  ██╔═══██╗██║   ██║██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗ ██║")
    print("  ██║   ██║██║   ██║█████╗  ███████╗   ██║   ██║██║   ██║██╔██╗██║")
    print("  ██║▄▄ ██║██║   ██║██╔══╝  ╚════██║   ██║   ██║██║   ██║██║╚████║")
    print("  ╚██████╔╝╚██████╔╝███████╗███████║   ██║   ██║╚██████╔╝██║ ╚███║")
    print("   ╚══▀▀═╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚══╝")
    print(f"{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}第{question_num}問{Colors.RESET}")
    print(f"{Colors.GREEN}Qwen vs Llama - どちらがどちらでしょう？{Colors.RESET}\n")


def display_llama_or_gemma_ascii():
    print(f"\n{Colors.BRIGHT_MAGENTA}")
    print("  ██╗     ██╗      █████╗ ███╗   ███╗ █████╗      ██████╗ ██████╗ ")
    print("  ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗    ██╔═══██╗██╔══██╗")
    print("  ██║     ██║     ███████║██╔████╔██║███████║    ██║   ██║██████╔╝")
    print("  ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║    ██║   ██║██╔══██╗")
    print("  ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║    ╚██████╔╝██║  ██║")
    print("  ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝     ╚═════╝ ╚═╝  ╚═╝")
    print("")
    print("   ██████╗ ███████╗███╗   ███╗███╗   ███╗ █████╗ ██████╗ ")
    print("  ██╔════╝ ██╔════╝████╗ ████║████╗ ████║██╔══██╗╚════██╗")
    print("  ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║███████║  ▄███╔╝")
    print("  ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══██║  ▀▀══╝ ")
    print("  ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║  ██╗   ")
    print("   ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝  ╚═╝   ")
    print(f"{Colors.RESET}")


def main():
    display_llama_or_gemma_ascii()

if __name__ == "__main__":
    main()