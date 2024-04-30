from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
)


def get_prompt(instruction: str) -> str:
    system = "Please answer the following question in a short and concise way."
    prompt = f"\n{instruction}"
    print(f"Question asked: {prompt}")
    return prompt


question = "Which city is the capital of Colombia?"
prompt = get_prompt(question)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
