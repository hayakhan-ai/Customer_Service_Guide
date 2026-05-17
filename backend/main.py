from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from rag import get_response, llm, retriever, prompt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(query: Query):
    answer = get_response(query.question)
    return {"answer": answer}


@app.post("/chat-stream")
def chat_stream(query: Query):

    docs = retriever.invoke(query.question)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(
        context=context,
        question=query.question
    )

    def stream():
       for chunk in llm.stream(final_prompt):
           text = getattr(chunk, "content", "")
           yield text.encode("utf-8")

    return StreamingResponse(stream(), media_type="text/plain")