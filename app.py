from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import nest_asyncio
from pyngrok import conf, ngrok
import uvicorn

import os
from dotenv import load_dotenv

# .env 파일 불러오기
load_dotenv()

# 모델 로드
base_model = "morirokim/ulsan_ai_tour_model"

baseModel = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    # device_map={"": 0}
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# pad_token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# pipeline 생성
pipe = pipeline(
    "text-generation",
    model=baseModel,
    tokenizer=tokenizer,
    # device=0  # CUDA 사용
)

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 형식 정의
class Question(BaseModel):
    question: str

# LLaMA3 응답 함수
def extract_response_llama3(instruction, input_text="", system_message="너는 울산 관광지 전문가야. 항상 답변 마지막에 '감사합니다' 라고 말해야해."):
    user_content = instruction
    if input_text:
        user_content += f"\n{input_text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 🛠️ eos_token과 eot_token을 int로 변환
    eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    try:
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_token_id, str):  # 안전성 체크
            eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    except:
        eot_token_id = eos_token_id  # fallback

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=[eos_token_id, eot_token_id],
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        num_return_sequences=1,
        return_full_text=False
    )

    generated_text = outputs[0]["generated_text"].strip()
    return {"response": generated_text}

# 챗봇 엔드포인트
@app.post("/chat", status_code=200)
async def chat_with_bot(x: Question):
    print(x)
    question = x.question  # 여기에 .question 빠져있었음
    response = extract_response_llama3(question)
    print(response)
    return {"response": response}

# ngrok 세팅
conf.get_default().auth_token = os.getenv("NGROK_AUTHTOKEN")
nest_asyncio.apply()
ngrok_tunnel = ngrok.connect(8001)
print(f"🌐 Public URL: {ngrok_tunnel.public_url}")

# 서버 실행
uvicorn.run(app, port=8001)