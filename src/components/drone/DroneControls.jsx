
import { useState } from 'react';

export default function DroneControls({ 
  isArmed, 
  droneState, 
  connectionStatus,
  onArmDisarm
}) {
  return (
    <div>
      <h3 className="text-lg font-semibold text-gray-700 mb-2">빠른 실행</h3>
      <div className="border-b mb-3"></div>
      <div className="flex space-x-2">
        <button 
          onClick={onArmDisarm}
          disabled={connectionStatus !== 'connected'}
          className={`
            flex items-center justify-center px-4 py-1.5 rounded-md text-sm font-medium
            ${isArmed 
              ? 'bg-green-500 text-white hover:bg-green-600' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}
            disabled:opacity-50 disabled:cursor-not-allowed
          `}
        >
          ✓ ARM
        </button>
        <button 
          disabled={connectionStatus !== 'connected' || !isArmed}
          className="flex items-center justify-center px-4 py-1.5 bg-gray-500 text-white rounded-md text-sm font-medium hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          🚀 이륙
        </button>
        <button 
          disabled={connectionStatus !== 'connected' || !droneState.isFlying}
          className="flex items-center justify-center px-4 py-1.5 bg-gray-600 text-white rounded-md text-sm font-medium hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          🛬 착륙
        </button>
      </div>
    </div>
  );
}
