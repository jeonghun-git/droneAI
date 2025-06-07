# 드론 센서 데이터 UI 연동 가이드

## 1. 개요
이 문서는 DroneVision AI Control UI에 실제 드론 데이터를 연동하는 방법을 설명합니다.

## 2. 연동 방법

### 2.1 WebSocket 연결 설정
- src/services/droneService.js 파일 생성
- WebSocket을 통해 드론 서버와 통신
- 연결 상태 관리 (connected, connecting, disconnected)

### 2.2 데이터 수신 및 상태 업데이트
- App.jsx의 useState 훅에서 관리하는 상태를 실제 데이터로 업데이트
- 주요 상태: connectionStatus, batteryLevel, isArmed, droneState, cameraState

### 2.3 데이터 형식
```javascript
// 드론 상태 데이터 형식
{
  isArmed: Boolean,
  isFlying: Boolean,
  altitude: Number,
  speed: Number,
  batteryLevel: Number,
  gps: {
    lat: Number,
    lng: Number,
    satellites: Number
  }
}

// 카메라 상태 데이터 형식
{
  streaming: Boolean,
  recording: Boolean,
  mode: String, // 'normal' 또는 'tracking'
  resolution: String // '720p', '1080p', '4K'
}
```

### 2.4 명령 전송
- 명령어 형식 정의 및 WebSocket을 통한 전송
- 주요 명령: ARM/DISARM, 이륙, 착륙, 카메라 제어

## 3. 구현 단계

1. WebSocket 서비스 구현
2. App.jsx에서 WebSocket 연결 관리
3. 상태 업데이트 함수 연결
4. 명령 전송 함수 구현
5. 에러 처리 및 재연결 로직 추가

## 4. 예시 코드

```javascript
// WebSocket 연결 설정
const connectDrone = async () => {
  try {
    setConnectionStatus('connecting');
    await droneService.connect('ws://드론서버주소:포트');
    setConnectionStatus('connected');
  } catch (error) {
    setConnectionStatus('disconnected');
    console.error('드론 연결 실패:', error);
  }
};

// 데이터 수신 처리
useEffect(() => {
  const handleDroneData = (data) => {
    if (data.batteryLevel !== undefined) setBatteryLevel(data.batteryLevel);
    if (data.isArmed !== undefined) setIsArmed(data.isArmed);
    if (data.droneState) setDroneState(data.droneState);
    if (data.cameraState) setCameraState(data.cameraState);
  };

  droneService.on('data', handleDroneData);
  return () => droneService.off('data', handleDroneData);
}, []);

// 명령 전송
const sendCommand = (command) => {
  try {
    droneService.sendCommand({
      type: 'command',
      command: command
    });
  } catch (error) {
    console.error('명령 전송 실패:', error);
  }
};
```

## 5. 필요한 백엔드 구성

- Node.js 서버 + WebSocket
- 드론 통신 프로토콜 구현 (MAVLink, DJI SDK 등)
- 데이터 변환 및 전송 로직

## 6. 테스트 방법

1. 모의 데이터로 테스트 (드론 시뮬레이터 활용)
2. 실제 드론 연결 테스트
3. 각 기능별 테스트 (연결, 명령 전송, 데이터 수신)
