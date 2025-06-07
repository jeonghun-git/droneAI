
import { useState, useEffect } from 'react';
import Header from './components/ui/Header';
import VoiceControl from './components/drone/VoiceControl';
import DroneControls from './components/drone/DroneControls';
import CameraControls from './components/camera/CameraControls';
import SystemLog from './components/ui/SystemLog';
import KakaoMap from './components/map/KakaoMap';

// Custom Hooks
import { useDroneState } from './hooks/useDroneState';
import { useCameraState } from './hooks/useCameraState';
import { useVoiceControl } from './hooks/useVoiceControl';

export default function DroneControlUI() {
  // Custom Hook을 통한 상태 관리
  const drone = useDroneState();
  const camera = useCameraState(drone.connectionStatus);
  const voice = useVoiceControl(drone.connectionStatus, drone.updateDroneFromCommand);

  // 연결 상태 변경 시 카메라 상태 초기화
  const handleConnect = () => {
    if (drone.connectionStatus === 'connected') {
      camera.resetCameraState();
    }
    drone.handleConnect();
  };

  return (
    <div className="min-h-screen bg-gray-300 text-gray-800 font-sans">
      {/* 헤더 */}
      <Header 
        connectionStatus={drone.connectionStatus} 
        onConnect={handleConnect} 
      />

      {/* 메인 컨텐츠 */}
      <main className="container mx-auto p-4 md:p-5 grid grid-cols-1 gap-4 md:gap-5">
        {/* 첫 번째 행: 명령 제어 패널과 카메라 제어 패널 */}
        <div className="grid grid-cols-1 mb-3 md:grid-cols-5 gap-4 md:gap-5">
          {/* 왼쪽 두 개의 컨테이너 */}
          <div className="md:col-span-2 flex flex-col gap-4 h-full">
            {/* 음성 명령 제어 컨테이너 */}
            {voice.isVoiceMode ? (
              <VoiceControl 
                isVoiceMode={voice.isVoiceMode}
                isRecording={voice.isRecording}
                autoVoice={voice.autoVoice}
                commandInput={voice.commandInput}
                connectionStatus={drone.connectionStatus}
                onToggleRecording={voice.toggleRecording}
                onToggleAutoVoice={voice.toggleAutoVoice}
                onCommandInputChange={(e) => voice.handleCommandInputChange(e.target.value)}
                onSendCommand={voice.handleSendCommand}
              />
            ) : (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <label htmlFor="commandInput" className="block text-xl font-semibold text-gray-700 mb-4">명령</label>
                <div className="flex gap-0">
                  <input
                    id="commandInput"
                    type="text"
                    value={voice.commandInput}
                    onChange={(e) => voice.handleCommandInputChange(e.target.value)}
                    className="flex-1 p-2 border border-gray-300 rounded-l-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm bg-white"
                    placeholder="명령을 직접 입력하세요..."
                    onKeyDown={(e) => e.key === 'Enter' && voice.handleSendCommand()}
                  />
                  <button
                    onClick={voice.handleSendCommand}
                    disabled={!voice.commandInput.trim() || drone.connectionStatus !== 'connected'}
                    className="bg-blue-500 text-white px-4 py-2 rounded-r-md hover:bg-blue-600 text-sm font-medium disabled:bg-gray-300 disabled:cursor-not-allowed whitespace-nowrap min-w-[60px]"
                  >
                    전송
                  </button>
                </div>
              </div>
            )}

            {/* 빠른 실행 컨테이너 */}
            <div className='bg-white rounded-xl shadow-lg p-6 flex-1'>
              <DroneControls 
                isArmed={drone.isArmed}
                droneState={drone.droneState}
                connectionStatus={drone.connectionStatus}
                onArmDisarm={drone.handleArmDisarm}
              />
            </div>
          </div>

          {/* 카메라 및 드론 상태 패널 */}
          <div className="md:col-span-3 h-full">
            <CameraControls 
              cameraState={camera.cameraState}
              connectionStatus={drone.connectionStatus}
              onToggleStreaming={camera.toggleStreaming}
              onToggleRecording={camera.toggleRecording}
              onChangeCameraMode={camera.changeCameraMode}
              onChangeCameraResolution={(e) => camera.changeCameraResolution(e.target.value)}
              droneState={drone.droneState}
            />
          </div>
        </div>
        
        {/* 두 번째 행: 지도와 시스템 로그 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-5">
          {/* 지도 컴포넌트 */}
          <div className="bg-white rounded-xl shadow-lg p-6 flex flex-col" style={{ minHeight: '300px', zIndex: 0 }}>
            <h2 className="text-xl font-semibold text-gray-700 border-b pb-2 mb-4">
              현재 위치
            </h2>
            <div style={{ height: '250px', position: 'relative', zIndex: 0 }}>
              <KakaoMap />
            </div>
          </div>

          {/* 시스템 콘솔 */}
          <div style={{ minHeight: '300px' }}>
            <SystemLog
              connectionStatus={drone.connectionStatus}
              isVoiceMode={voice.isVoiceMode}
              autoVoice={voice.autoVoice}
              isArmed={drone.isArmed}
              droneState={drone.droneState}
              isRecording={voice.isRecording}
              commandInput={voice.commandInput}
              cameraState={camera.cameraState}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
