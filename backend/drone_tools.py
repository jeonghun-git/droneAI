from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pymavlink import mavutil
import asyncio
from typing import Dict, Any
import json

app = FastAPI()

# MAVLink 연결 설정
def connect_to_pixhawk():
    try:
        conn = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
        conn.wait_heartbeat(timeout=5)
        print("Pixhawk 연결 성공!")
        return conn
    except Exception as e:
        print(f"Pixhawk 연결 실패: {e}")
        return None

conn = connect_to_pixhawk()

# 연결 관리 클래스
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data: Dict[str, Any]):
        for connection in self.active_connections:
            await connection.send_json(data)

manager = ConnectionManager()

# Tool Calling을 위한 함수 정의
def get_sensor_data(sensor_type: str) -> Dict[str, Any]:
    """사용자가 요청한 센서 데이터 반환"""
    if not conn:
        return {"error": "Pixhawk not connected"}
    
    try:
        if sensor_type == "gps":
            msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            return {
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "alt": msg.alt / 1000  # m 단위
            }
        elif sensor_type == "battery":
            msg = conn.recv_match(type='SYS_STATUS', blocking=True)
            return {
                "remaining": msg.battery_remaining,
                "voltage": msg.voltage_battery / 1000  # V 단위
            }
        elif sensor_type == "attitude":
            msg = conn.recv_match(type='ATTITUDE', blocking=True)
            return {
                "roll": msg.roll,
                "pitch": msg.pitch,
                "yaw": msg.yaw
            }
        elif sensor_type == "velocity":
            msg = conn.recv_match(type='VFR_HUD', blocking=True)
            return {
                "speed": msg.ground_speed,
                "heading": msg.heading
            }
        else:
            return {"error": "Invalid sensor type"}
    except Exception as e:
        return {"error": str(e)}

def drone_control(action: str, value: float = None) -> Dict[str, Any]:
    """드론 제어 명령 실행"""
    if not conn:
        return {"error": "Pixhawk not connected"}
    
    try:
        if action == "takeoff":
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 5  # 고도 5m
            )
            return {"status": "takeoff initiated"}
        elif action == "land":
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0
            )
            return {"status": "landing initiated"}
        elif action == "move":
            direction_map = {
                "forward": mavutil.mavlink.MAV_FRAME_BODY_NED,
                "backward": mavutil.mavlink.MAV_FRAME_BODY_NED,
                "left": mavutil.mavlink.MAV_FRAME_BODY_NED,
                "right": mavutil.mavlink.MAV_FRAME_BODY_NED,
                "up": mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                "down": mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            }
            conn.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                0, conn.target_system, conn.target_component,
                direction_map[action], 0b110111111000, 0, 0, 0,
                value if action in ["forward", "up"] else -value if action in ["backward", "down"] else 0,
                value if action == "right" else -value if action == "left" else 0,
                0, 0, 0, 0, 0
            ))
            return {"status": f"moving {action} by {value}m"}
        elif action == "rotate":
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, value, 0, 0, 0, 0, 0, 0
            )
            return {"status": f"rotating by {value} degrees"}
        else:
            return {"error": "Invalid action"}
    except Exception as e:
        return {"error": str(e)}

# 실시간 데이터 브로드캐스팅
async def telemetry_task():
    while conn:
        try:
            data = {
                "gps": get_sensor_data("gps"),
                "battery": get_sensor_data("battery"),
                "attitude": get_sensor_data("attitude"),
                "velocity": get_sensor_data("velocity")
            }
            await manager.broadcast(data)
        except Exception as e:
            print(f"Error in telemetry: {e}")
        await asyncio.sleep(0.1)

# FastAPI Startup Event
@app.on_event("startup")
async def startup_event():
    if conn:
        asyncio.create_task(telemetry_task())

# 웹소켓을 통한 실시간 데이터 전송
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Tool Calling을 위한 API 엔드포인트
@app.post("/get_sensor_data")
async def api_get_sensor_data(sensor_type: str):
    return get_sensor_data(sensor_type)

@app.post("/drone_control")
async def api_drone_control(action: str, value: float = None):
    return drone_control(action, value)