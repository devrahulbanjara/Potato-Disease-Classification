from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message": "Jinda hu bhay !"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")
