from ctransformers import AutoModelForCausalLM

model_path = ("../models/models--TheBloke--Llama-2-7B-GGUF/"
              "snapshots/b4e04e128f421c93a5f1e34ac4d7ca9b0af47b80")

llm = AutoModelForCausalLM.from_pretrained(
                                            model_path,
                                            # "TheBloke/Llama-2-7B-GGUF",
                                            model_file="llama-2-7b.Q4_K_M.gguf",
                                            model_type="llama", gpu_layers=50)
# User interface
prompt = "Which is the largest country in the world by population?"
print(llm(f"Question: {prompt} Answer:", stop=["\n", "Question:", "Q:"]))