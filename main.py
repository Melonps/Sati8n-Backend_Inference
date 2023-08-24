from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel  # リクエストbodyを定義するために必要
import lightGBM_Inference

app = FastAPI()

origins = ["http://localhost:3000", "http://localhost", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = lightGBM_Inference.StomachPredictionModel()


class PostData(BaseModel):
    age: int
    sex: str
    height: int
    weight: int
    now_time: int
    free_time: int
    had_calory: int
    eat_calory: int


@app.get("/")
async def get():
    return "Hello World"


@app.post("/inference/")
async def update_user(data: PostData):
    try:
        get_data = {
            "年齢": data.age,
            "性別": data.sex,
            "身長": data.height,
            "体重": data.weight,
            "今の時間帯": data.now_time,
            "空いた時間": data.free_time,
            "食べたカロリー": data.had_calory,
            "これから食べるカロリー": data.eat_calory,
        }
        prediction = model.predict(get_data)
        return {"prediction": prediction}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="推論中にエラーが発生しました")
