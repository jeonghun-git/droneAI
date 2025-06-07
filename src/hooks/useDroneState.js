import { useState } from 'react';

export const useDroneState = () => {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [batteryLevel, setBatteryLevel] = useState(0);
  const [isArmed, setIsArmed] = useState(false);
  const [droneState, setDroneState] = useState({
    gps: { lat: 37.5665, lng: 126.9780, satellites: 12 },
    altitude: 0,
    speed: 0,
    isFlying: false
  });

  const handleConnect = () => {
    if (connectionStatus === 'connected') {
      setConnectionStatus('disconnected');
      setBatteryLevel(0);
      setIsArmed(false);
      setDroneState(prev => ({ ...prev, isFlying: false, altitude: 0, speed: 0 }));
    } else {
      setConnectionStatus('connecting');
      setTimeout(() => {
        setConnectionStatus('connected');
        setBatteryLevel(87);
      }, 1500);
    }
  };

  const handleArmDisarm = () => {
    if (connectionStatus !== 'connected') return;
    if (droneState.isFlying && isArmed) {
      alert("비행 중에는 Disarm 할 수 없습니다.");
      return;
    }
    setIsArmed(!isArmed);
  };

  const updateDroneFromCommand = (command) => {
    if (command.toLowerCase().includes('이륙')) {
      setDroneState(prev => ({ ...prev, isFlying: true, altitude: 10, speed: 5 }));
    } else if (command.toLowerCase().includes('착륙')) {
      setDroneState(prev => ({ ...prev, isFlying: false, altitude: 0, speed: 0 }));
    } else if (command.toLowerCase().includes('고도')) {
      const altMatch = command.match(/고도 (\d+)/);
      if (altMatch && altMatch[1]) {
        setDroneState(prev => ({ ...prev, altitude: parseInt(altMatch[1]) }));
      }
    }
  };

  return {
    connectionStatus,
    batteryLevel,
    isArmed,
    droneState,
    handleConnect,
    handleArmDisarm,
    updateDroneFromCommand,
    setDroneState,
    setBatteryLevel,
    setConnectionStatus
  };
}; 