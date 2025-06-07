import { useState } from 'react';

export default function VoiceControl({ 
  isVoiceMode, 
  isRecording, 
  autoVoice, 
  commandInput, 
  connectionStatus,
  onToggleRecording, 
  onToggleAutoVoice, 
  onCommandInputChange,
  onSendCommand
}) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-700 mb-4">음성 명령 제어</h2>
      <div className="border-b pb-2 mb-4  ">
        <div className="flex space-x-2 mb-4">
          <button 
            onClick={onToggleRecording}
            disabled={!isVoiceMode || connectionStatus !== 'connected'}
            className={`
              flex items-center justify-center px-4 py-1.5 rounded-md text-sm font-medium
              ${isRecording ? 'bg-red-100 text-red-600 animate-pulse' : 'bg-gray-100 text-gray-600'}
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {isRecording ? '🛑 녹음 중지' : '🎙️ 음성 녹음'}
          </button>
          <button 
            onClick={onToggleAutoVoice}
            disabled={!isVoiceMode}
            className={`
              flex items-center justify-center px-4 py-1.5 rounded-md text-sm font-medium
              ${autoVoice ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-600'}
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {autoVoice ? '🔄 자동 ON' : '🔄 자동 OFF'}
          </button>
        </div>
      </div>

      <div className="mb-4">
        <label className="block text-md font-semibold text-gray-700 mb-2 ml-1">명령</label>
        <div className="flex">
          <input 
            type="text" 
            value={commandInput}
            onChange={onCommandInputChange}
            placeholder="음성 녹음 버튼을 누르세요"
            disabled={isRecording || connectionStatus !== 'connected'}
            className="flex-1 py-2 px-3 border border-gray-300 bg-white rounded-l-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
          />
          <button 
            onClick={onSendCommand}
            disabled={!commandInput.trim() || connectionStatus !== 'connected'}
            className="bg-gray-600 text-white py-2 px-4 rounded-r-md font-medium hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            전송
          </button>
        </div>
      </div>
    </div>
  );
}
