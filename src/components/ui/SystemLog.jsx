
import { useState } from 'react';

export default function SystemLog({ 
  connectionStatus, 
  isVoiceMode, 
  autoVoice, 
  isArmed, 
  droneState,
  isRecording,
  commandInput,
  cameraState
}) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 col-span-3 flex flex-col">
      <div className="flex justify-between items-center border-b pb-2 mb-4">
        <h2 className="text-xl font-semibold text-gray-700">📜 시스템 로그</h2>
        <div className="text-xs text-gray-500">{new Date().toLocaleString()}</div>
      </div>
      <div className="bg-gray-800 text-gray-300 font-mono text-xs p-4 rounded-md h-48 overflow-y-auto leading-relaxed">
        <p><span className="text-blue-400">$ DroneAgent AI Control v1.0 initialized.</span></p>
        <p><span className={connectionStatus === 'connected' ? 'text-green-400' : connectionStatus === 'connecting' ? 'text-yellow-400 animate-pulse' : 'text-red-400'}>$ Connection: {connectionStatus.toUpperCase()}</span></p>
        <p><span className="text-cyan-400">$ Mode: {isVoiceMode ? 'Voice Recognition' : 'Manual Text Input'}</span></p>
        {isVoiceMode && <p><span className="text-cyan-400">$ Voice Setting: {autoVoice ? 'Auto Transcription ON' : 'Auto Transcription OFF'}</span></p>}
        <p><span className="text-purple-400">$ Drone Status: {isArmed ? 'ARMED' : 'DISARMED'} | {droneState.isFlying ? `FLYING (Alt: ${droneState.altitude}m, Spd: ${droneState.speed}m/s)` : 'IDLE'}</span></p>
        {!isRecording && commandInput && <p><span className="text-lime-400">$ Last Command Sent: "{commandInput}"</span></p>}
        {isRecording && <p><span className="text-pink-400 animate-pulse">$ Voice input active...Listening...</span></p>}
        {cameraState.streaming && <p><span className="text-teal-400">$ Camera Stream: ACTIVE | Mode: {cameraState.mode} | Resolution: {cameraState.resolution}</span></p>}
        {cameraState.recording && <p><span className="text-red-400 animate-pulse">$ Camera Recording: ACTIVE</span></p>}
      </div>
    </div>
  );
}
