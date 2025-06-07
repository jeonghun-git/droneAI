
import { useState } from 'react';

import React from 'react';

export default function CameraControls({ 
  cameraState, 
  connectionStatus,
  onToggleStreaming,
  onToggleRecording,
  onChangeCameraMode,
  onChangeCameraResolution,
  droneState
}) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-4 md:p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-3 md:mb-4">
        <h2 className="text-lg md:text-xl font-semibold text-gray-700">📷 카메라 & 드론 상태</h2>
        <div className="text-xs md:text-sm bg-gray-100 px-2 md:px-3 py-1 rounded-full font-medium">
          🔋 {connectionStatus === 'connected' ? '87%' : '0%'}
        </div>
      </div>
      
      <div className="bg-black rounded-lg mb-3 md:mb-4 flex items-center justify-center flex-1" style={{ minHeight: '180px' }}>
        {connectionStatus === 'connected' ? (
          cameraState.streaming ? (
            <div className="relative w-full h-full">
              {/* 여기에 실제 카메라 스트림이 들어갈 것입니다 */}
              <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
                {/* 임시 카메라 화면 */}
                <div className="text-white text-lg">카메라 스트림 활성화됨</div>
              </div>
              {cameraState.recording && (
                <div className="absolute top-4 right-4 bg-red-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">
                  REC
                </div>
              )}
              {cameraState.mode === 'tracking' && (
                <div className="absolute top-4 left-4 bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                  추적 모드
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-400 flex flex-col items-center">
              <svg className="w-16 h-16 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span>드론 연결 필요</span>
            </div>
          )
        ) : (
          <div className="text-gray-400 flex flex-col items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-16 h-16 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            <span>드론 연결 필요</span>
          </div>
        )}
      </div>
      
      <div className="flex flex-col md:flex-row md:justify-between mb-3 md:mb-4 gap-2 md:gap-0">
        <div className="flex space-x-2">
          <button 
            onClick={onToggleStreaming}
            disabled={connectionStatus !== 'connected'}
            className={`
              py-2 px-3 md:py-2.5 md:px-4 rounded-md font-semibold text-xs focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors
              ${cameraState.streaming 
                ? 'bg-red-500 text-white hover:bg-red-600 focus:ring-red-500' 
                : 'bg-gray-400 text-white hover:bg-gray-600 focus:ring-gray-500'}
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {cameraState.streaming ? '🛑 스트림 정지' : '▶️ 스트림 시작'}
          </button>
          <button 
            onClick={onToggleRecording}
            disabled={connectionStatus !== 'connected' || !cameraState.streaming}
            className={`
              py-2 px-3 md:py-2.5 md:px-4 rounded-md font-semibold text-xs focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors
              ${cameraState.recording 
                ? 'bg-red-500 text-white hover:bg-red-600 focus:ring-red-500 animate-pulse' 
                : 'bg-gray-200 text-gray-600 hover:bg-gray-300 focus:ring-gray-500'}
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {cameraState.recording ? '⏹️ 녹화 정지' : '⏺️ 녹화 시작'}
          </button>
        </div>
        
        <div className="flex space-x-2 mt-2 md:mt-0">
          <button 
            onClick={onChangeCameraMode}
            disabled={connectionStatus !== 'connected'}
            className="
              py-2 px-3 md:py-2.5 md:px-4 bg-gray-200 text-gray-700 rounded-md font-semibold text-xs hover:bg-gray-300 
              focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors
              disabled:opacity-50 disabled:cursor-not-allowed
            "
          >
            {cameraState.mode === 'tracking' ? '추적 모드' : '일반 모드'}
          </button>
          <select 
            value={cameraState.resolution}
            onChange={onChangeCameraResolution}
            disabled={connectionStatus !== 'connected'}
            className="py-2 px-2 md:py-2.5 md:px-3 bg-gray-100 border border-gray-300 text-gray-700 rounded-md font-semibold text-xs focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="720p">HD (720p)</option>
            <option value="1080p">FHD (1080p)</option>
            <option value="4K">UHD (4K)</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 md:gap-4 text-center">
        <div className="bg-gray-100 p-2 md:p-3 rounded-lg">
          <div className="text-xs text-gray-500 uppercase tracking-wider">위치 (Lat, Lng)</div>
          <div className="font-semibold text-xs md:text-sm mt-1">
            {droneState.gps.lat.toFixed(4)}, {droneState.gps.lng.toFixed(4)}
          </div>
        </div>
        <div className="bg-gray-100 p-2 md:p-3 rounded-lg">
          <div className="text-xs text-gray-500 uppercase tracking-wider">고도 / 속도</div>
          <div className="font-semibold text-xs md:text-sm mt-1">
            {droneState.altitude.toFixed(1)}m / {droneState.speed.toFixed(1)}m/s
          </div>
        </div>
        <div className="bg-gray-100 p-2 md:p-3 rounded-lg">
          <div className="text-xs text-gray-500 uppercase tracking-wider">GPS 위성</div>
          <div className="font-semibold text-xs md:text-sm mt-1">
            {droneState.gps.satellites} <span className={droneState.gps.satellites > 6 ? 'text-green-500' : 'text-orange-500'}>({droneState.gps.satellites > 6 ? '안정적' : '불안정'})</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// 드론 상태 기본값 (props가 전달되지 않을 경우)
CameraControls.defaultProps = {
  droneState: {
    gps: { lat: 37.5665, lng: 126.9780, satellites: 12 },
    altitude: 0,
    speed: 0,
    isFlying: false
  }
};
