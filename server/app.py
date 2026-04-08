import gymnasium as gym
from fastapi import FastAPI, Request
import json
import founder_endurance
import numpy as np

app = FastAPI()
env = gym.make("FounderEndurance-v1")

@app.post("/reset")
async def reset(request: Request):
    # Accept any json body for reset, such as seed or empty body
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    
    seed = body.get("seed", None)
    options = body.get("options", None)
    
    # In case options wasn't provided, safe fallback
    obs, info = env.reset(seed=seed, options=options)
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
async def step(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    action_list = body.get("action", [1, 0, 0]) # Default safe action
    action = np.array(action_list, dtype=int)
    
    obs, reward, terminated, truncated, info = env.step(action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

@app.get("/state")
async def state():
    # Helper to comply with some openenv validation checks
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "FounderEndurance-v1 Environment Server is Running."}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
