import { useState } from 'react';

export const useCameraState = (connectionStatus) => {
  const [cameraState, setCameraState] = useState({
    streaming: false,
    recording: false,
    mode: 'normal', // normal / tracking
    resolution: '1080p'
  });

  const toggleStreaming = () => {
    if (connectionStatus !== 'connected') return;
    setCameraState(prev => ({ 
      ...prev, 
      streaming: !prev.streaming, 
      recording: prev.streaming ? false : prev.recording 
    }));
  };

  const toggleRecording = () => {
    if (connectionStatus !== 'connected' || !cameraState.streaming) return;
    setCameraState(prev => ({ ...prev, recording: !prev.recording }));
  };

  const changeCameraMode = () => {
    if (connectionStatus !== 'connected') return;
    setCameraState(prev => ({ 
      ...prev, 
      mode: prev.mode === 'normal' ? 'tracking' : 'normal' 
    }));
  };
  
  const changeCameraResolution = (resolution) => {
    setCameraState(prev => ({ ...prev, resolution }));
  };

  const resetCameraState = () => {
    setCameraState(prev => ({ 
      ...prev, 
      streaming: false, 
      recording: false 
    }));
  };

  return {
    cameraState,
    toggleStreaming,
    toggleRecording,
    changeCameraMode,
    changeCameraResolution,
    resetCameraState,
    setCameraState
  };
}; 