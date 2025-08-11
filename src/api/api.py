import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

app = FastAPI()

from src.config import TRACES_DIR


@app.get("/")
def home():
    return "Hello!"


@app.post("/otel/traces/{agent_name}")
async def write_otel_traces_to_local_file(
        *,
        agent_name: str | None = None,
        request: Request
):
    byte_data = await request.body()

    # Parse it
    msg = ExportTraceServiceRequest()
    msg.ParseFromString(byte_data)

    msg_dict = MessageToDict(msg, preserving_proto_field_name=True)

    file_name = datetime.now().isoformat() + "-trace.json"
    if agent_name:
        file_name = f"{agent_name}/{file_name}"
    full_path = os.path.join(TRACES_DIR, file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(os.path.join(TRACES_DIR, file_name), "w") as f:
        json.dump(msg_dict, f, indent=2)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=7000)
    # run with uvicorn src.api:app --reload
