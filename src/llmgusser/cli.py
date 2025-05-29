from llama_cpp import Llama
import sys
import os
import time
import random
import contextlib
from llmgusser.ascii_art import display_llama_or_gemma_ascii
from llmgusser.spinner import Spinner


is_model_load = False

HOME = os.environ["HOME"]

model_path_llama =  os.path.join(HOME, ".llmqa/models/llama-3.2-3b-instruct-q4_k_m.gguf")
model_path_gemma = os.path.join(HOME, ".llmqa/models/gemma-3-4b-it-Q4_K_M.gguf")


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield


# 以下を実装中
def load_model():
    with suppress_output():
        # llm_llama = Llama.from_pretrained(repo_id=model_path_llama, filename=model_filename_llama, n_ctx=1000, verbose=False, chat_format="llama-3")
        # llm_gemma = Llama.from_pretrained(repo_id=model_path_gemma, filename=model_filename_gemma, n_ctx=1000, verbose=False, chat_format="gemma")
        llm_llama = Llama(model_path=model_path_llama, verbose=False, chat_format="llama-3")
        llm_gemma = Llama(model_path=model_path_gemma, verbose=False, chat_format="gemma")
    return llm_llama, llm_gemma


def model_inference(llm, user_text):
    # スピナーメッセージを追加
    spinner = Spinner("thinking ")
    spinner.start()

    content = "あなたは日本語で自然に返答してください。"

    try:
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
            
            messages = [
                {"role": "system", "content": content},
                {"role": "user", "content": user_text},
            ]

            spinner_stopped = False

            for chunk in llm.create_chat_completion(
                messages=messages, 
                max_tokens=100, 
                stream=True,
                ):
                delta = chunk["choices"][0]["delta"]
                
                if not spinner_stopped:
                    if "content" in delta:
                        spinner.stop()
                        spinner_stopped = True
                
                if "content" in delta:
                    print(delta["content"], end="", flush=True)

            if not spinner_stopped:
                spinner.stop()
            print()

    except Exception as e:
        spinner.stop()
        print(f"エラーが発生しました: {e}")


def question1(llm_llama, llm_gemma):        
    print()
    print("モデルに文章を与え，その生成文を見て回答してください．")
    print("では，モデルに与える文章を入力してください．(1 ~ 30 文字)", end="\n\n")
    
    while True:
        print(">>> ", end="")
        user_text = input()
        if 0 < len(user_text) < 31:
            break
        print("入力する文章の長さは，1 ~ 30 文字にしてください．", end="\n\n")

    correct_answer = random.randint(1, 2) # 1 == Llama
    
    for i, name in enumerate(["A", "B"]):
        print()
        if correct_answer== 1:
            if i == 0:
                print(f"LLM {name}:")
                model_inference(llm=llm_llama, user_text=user_text)
            else:
                print(f"LLM {name}:")
                model_inference(llm=llm_gemma, user_text=user_text)
        elif correct_answer == 2:
            if i == 0:
                print(f"LLM {name}:")
                model_inference(llm=llm_gemma, user_text=user_text)
            else:
                print(f"LLM {name}:")
                model_inference(llm=llm_llama, user_text=user_text)
    
    print("")
    
    which_answer = random.randint(1, 2) # 1 == Llama
    if which_answer == 1:
        which_answer_model = "Llama"
    else:
        which_answer_model = "Gemma"
        
    print(which_answer_model + " だと思う方の記号を入力してください．(A or B)", end="\n\n")
    while True:
        print(">>> ", end="")
        user_answer = input()
        if user_answer.lower() in ["a", "b"]:
            break
        else:
            print("解答形式が間違っています．A or B で解答してください．", end="\n\n")

    if which_answer_model == "Llama":
        # Llamaの場合：A=1, B=2
        user_answer_num = 1 if user_answer == "A" else 2
        correct_model = "Llama"
        wrong_model = "Gemma"
    else:
        # Gemmaの場合：A=2, B=1（選択肢が逆転）
        user_answer_num = 2 if user_answer == "A" else 1
        correct_model = "Gemma"
        wrong_model = "Llama"

    # 正解判定
    if user_answer_num == correct_answer:
        print(f"正解です！ {user_answer} が{correct_model}でした！")
    else:
        # 不正解の場合、正解の選択肢を表示
        if which_answer_model == "Llama":
            # Llamaの場合：正解が1なら"A"、2なら"B"
            correct_choice = "A" if correct_answer == 1 else "B"
            print(f"不正解です...正解は {correct_choice} ({correct_model}) です。")
        else:
            # Gemmaの場合：正解が1なら"B"、2なら"A"（逆転）
            correct_choice = "B" if correct_answer == 1 else "A"
            print(f"不正解です...正解は {correct_choice} ({correct_model}) です。")



def main():
    try:
        display_llama_or_gemma_ascii()
        print("LLM 当てクイズ")
        print("モデルは，Llama or Gemma の2択です．")
        print("終了する場合は ctrl+c を入力してください．")
        
        spinner_loading = Spinner("model loading ")
        spinner_loading.start()
        llm_llama, llm_gemma = load_model()
        spinner_loading.stop()

        # 1問目
        question1(llm_llama, llm_gemma)

        print("\n")

    except KeyboardInterrupt:
        print("\n")
        sys.exit()


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    main()
