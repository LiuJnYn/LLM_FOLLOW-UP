import os
import json
import dashscope
from http import HTTPStatus
import time
from typing import Dict, Any
import matplotlib.pyplot as plt # 绘制图像
from form_data.form_setting.long_complex_form import FORM_PROMPT_LONG_COMPLEX
from form_data.form_setting.short_form import FORM_PROMPT_SHORT
from form_data.form_setting.long_form import FORM_PROMPT_LONG
from form_data.patient_setting.patient1 import PATIENT_1,PATIENT_2,PATIENT_3


dashscope.api_key = ""
MODEL_NAME = "qwen-plus"
FORM_TYPE = "short_form"  # "short_form" 或 "long_complex_form"
PATIENT_TYPE = "3"

class UnstableAutonomousBot:
    def __init__(self, form_prompt: str):
        self.form_prompt = form_prompt
        self.conversation_history = []
        self.collected_answers = {}
        self.is_complete = False

        self.total_tokens = 0
        self.total_time = 0.0
        self.turn_count = 0
        self.elapsed_times = []
        self.turn_tokens = []
        self.turn_logs = []

    def save_turn_logs(self):
        """保存每轮对话日志和最终统计摘要到 JSON 文件。"""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"dialogue_turns_{timestamp}.json"
        output_dir = f"long_form/experiment_results/turn_logs/black/{FORM_TYPE}/{PATIENT_TYPE}"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        save_data = {
            "turn_logs": self.turn_logs, 
            "summary": {                   
                "总问答轮数": self.turn_count,
                "总耗时(秒)": round(self.total_time, 2),
                "平均响应时间(秒/轮)": round(self.total_time / self.turn_count, 2) if self.turn_count else 0,
                "累计token消耗": self.total_tokens
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)

        print(f"[结果] 对话回合日志及统计摘要已保存至: {filepath}")

    def simulate_patient_response(self, question: str) -> str:
        """
        模拟患者回答。
        """

        patient_profile = f"""
        你现在是一名患者，需要对医生提出的问题进行回答。
        这是你的性格特质和对话风格：
        {PATIENT_1 if PATIENT_TYPE == "1" else PATIENT_2 if PATIENT_TYPE == "2" else PATIENT_3}
        """
        prompt = f"""
        {patient_profile}

        这是你需要回答的新问题：“{question}”
        请做出回答：
        """
        response = dashscope.Generation.call(
            MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )
        if response and response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content.strip()
        else:
            return "我不太清楚。"

    def process_next_turn(self) -> Dict[str, Any]:
        history_str = "\n".join([f"医生助手: {turn['question']}\n患者: {turn['answer']}" for turn in self.conversation_history])
        
        system_prompt = """
        你是一位临床随访助手。你的任务是引导患者完成一份问卷。
        你必须严格遵循【工作流程与思维链】中的所有指令。
        你的最终输出必须且只能是一个纯粹的JSON对象。
        """
        
        user_prompt = f"""
        【完整问卷内容与对话内容】
        {self.form_prompt}
        {history_str if history_str else "无历史记录，请从头开始提问。"}

        ---
        【患者的最新回答】
        {self.conversation_history[-1]['answer'] if self.conversation_history else "无，请开始第一次提问。"}

        ---

        【工作流程与思维链】
        1. **阅读并理解**: 首先，仔细阅读下面提供的【完整问卷内容】，理解所有问题、选项和跳转逻辑。
        2. **分析与提取**: 分析【患者的最新回答】，将其内容与【完整问卷内容】中的选项进行匹配，提取患者回复中对应的选项。注意：选项只能限制在题目对应的选项之中。需要输出患者选中选项的内容。如果题目是填空题，则提取患者回答中的关键信息作为答案。如果没提取到患者的回答，需要进行追问，以获取准确的答案。
        3. **决策与生成**: 基于提取的答案和问卷逻辑，决定下一个问题，并将其进行语句上的修饰，生成为一句自然问话。请作为医生与患者交互，提出的问题需亲切、专业、简洁，让患者能够容易理解。选项仅作为提问策略的参考，可以不直接包含在问题中。问题长度需控制在30字以内。保存至"natural_next_question"字段中。
        4. **判断与报告**: 判断问卷是否完成，并按指定的JSON格式返回你的所有工作成果。
        
        【你的任务】
        请严格按照上述【工作流程与思维链】执行任务，并返回下方指定的JSON对象。
        `{{"last_question_answered_raw": "刚才处理的那个原始问题的文本", "extracted_answer": ["标准化答案1"], "natural_next_question": "你生成的人性化下一句问话", "is_complete": false}}`
        """

        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
        ]
        
        start_time = time.time()
        response = dashscope.Generation.call(
            MODEL_NAME,
            messages=messages,
            result_format='message',
        )
        end_time = time.time()
        
        elapsed = end_time - start_time
        self.total_time += elapsed
        self.turn_count += 1
        self.elapsed_times.append(elapsed)

        if response and response.status_code == HTTPStatus.OK:
            usage = getattr(response, "usage", None)
            current_turn_tokens = 0 # 初始化本轮token为0
            if usage and "total_tokens" in usage:
                current_turn_tokens = usage["total_tokens"] # 获取本轮消耗
                self.total_tokens += current_turn_tokens
            self.turn_tokens.append(current_turn_tokens)

            content = response.output.choices[0].message.content
            if content.strip().startswith("```json"):
                content = content.strip()[7:-3]
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"--- [错误] JSON解析失败: {e} --- \n[原始输出]: {content}")
                return None
        else:
            print(f"--- [错误] API调用失败: {response}")
            return None

    def save_and_plot_results(self):
        """在对话结束后，保存响应时间数据并绘制图表。"""
        # --- 1. 保存响应时间到JSON文件 ---
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"response_times_{timestamp}.json"
        
        # 确保保存的目录存在
        output_dir = f"long_form/experiment_results/time_data/black/{FORM_TYPE}/{PATIENT_TYPE}"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.elapsed_times, f, ensure_ascii=False, indent=4)
        print(f"\n[结果] 响应时间数据已保存至: {filepath}")

        # --- 2. 绘制消耗token变化曲线图 ---
        if self.turn_tokens: # <--- 修改：检查 turn_tokens
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            plt.figure(figsize=(12, 6))
            # <--- 修改：绘制 turn_tokens
            plt.plot(range(1, self.turn_count + 1), self.turn_tokens, marker='s', linestyle='--')
            plt.title('Token Consumption per Dialogue Turn', fontsize=16)
            plt.xlabel('Turn', fontsize=12)
            plt.ylabel('Tokens Consumed', fontsize=12)
            plt.xticks(range(1, self.turn_count + 1))
            plt.grid(True)

            timestamp_img = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename_img = f"token_consumption_plot_{timestamp_img}.png" # 可以改为 .jpg, .pdf 等
            output_dir_img = f"long_form/experiment_results/plots/black/{FORM_TYPE}/{PATIENT_TYPE}" # 指定图片保存目录
            os.makedirs(output_dir_img, exist_ok=True) # 确保目录存在
            filepath_img = os.path.join(output_dir_img, filename_img)
            plt.savefig(filepath_img, dpi=300) # dpi设置图片分辨率
            print(f"[结果] Token消耗图表已保存至: {filepath_img}")

    def start(self):
        print("您好！我是您的随访助手，现在我们开始进行沟通。")
        
        initial_result = self.process_next_turn()
        if not initial_result:
            print("系统启动失败，请检查API设置。")
            return
        
        result = initial_result
        timer = time.time()
        
        while not self.is_complete:
            if self.turn_count > 80:
                print("[医生助手] 抱歉，问卷问答轮次过多，系统自动终止此次随访。")
                print("\n==============================================")
                print("随访完成，感谢您的配合！以下为统计摘要：")
                print(f"总问答轮数: {self.turn_count}")
                print(f"总耗时: {self.total_time:.2f} 秒")
                if self.turn_count > 0:
                    print(f"平均响应时间: {self.total_time / (self.turn_count-1):.2f} 秒/轮")
                print(f"累计 token 消耗: {self.total_tokens} tokens")
                print("==============================================")

                self.save_turn_logs()
                self.save_and_plot_results()
                break
            natural_question = result.get("natural_next_question")
            print(f"\n[医生助手] {natural_question}")
            print(f"\n[生成问题所用时间] {time.time() - timer:.2f} 秒")

            user_answer = self.simulate_patient_response(natural_question)
            print(f"患者: {user_answer}")
            
            self.conversation_history.append({
                "question": natural_question,
                "answer": user_answer
            })
            
            timer = time.time() # 重置计时器
            result = self.process_next_turn()

            if not result:
                print("[医生助手] 抱歉，系统似乎出了一点小问题，我们稍后重试。")
                break
            
            last_q_raw = result.get("last_question_answered_raw")
            extracted = result.get("extracted_answer")
            next_question = result.get("natural_next_question")

            turn_elapsed = self.elapsed_times[-1] if self.elapsed_times else None
            self.turn_logs.append({
                "natural_question": natural_question,
                "elapsed": turn_elapsed,
                "user_answer": user_answer,
                "extracted_answer": extracted,
                "natural_next_question": next_question
            })

            if last_q_raw and extracted:
                self.collected_answers[last_q_raw] = extracted
                print(f"--- [后台记录] {extracted} ---")

            self.is_complete = result.get("is_complete", False)
            
        print("\n==============================================")
        print("随访完成，感谢您的配合！以下为统计摘要：")
        print(f"总问答轮数: {self.turn_count}")
        print(f"总耗时: {self.total_time:.2f} 秒")
        if self.turn_count > 0:
            print(f"平均响应时间: {self.total_time / (self.turn_count-1):.2f} 秒/轮")
        print(f"累计 token 消耗: {self.total_tokens} tokens")
        print("==============================================")

        self.save_turn_logs()
        self.save_and_plot_results()

if __name__ == "__main__":
    for i in range():
        print(f"\n\n================= 运行第 {i+1} 次 =================\n\n")
        bot = UnstableAutonomousBot(FORM_PROMPT_SHORT if FORM_TYPE == "short_form" else FORM_PROMPT_LONG if FORM_TYPE == "long_form" else FORM_PROMPT_LONG_COMPLEX)
        bot.start()