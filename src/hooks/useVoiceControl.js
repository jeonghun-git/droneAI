import { useState } from 'react';

export const useVoiceControl = (connectionStatus, onDroneCommand) => {
  const [isVoiceMode, setIsVoiceMode] = useState(true);
  const [commandInput, setCommandInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [autoVoice, setAutoVoice] = useState(true);

  const toggleRecording = () => {
    if (!isVoiceMode || connectionStatus !== 'connected') return;
    
    setIsRecording(prev => !prev); 
    if (!isRecording) { 
      // 녹음 시작
      setCommandInput('');
      setTimeout(() => {
        setCommandInput(autoVoice ? '명령 입력: 고도 30미터 상승' : '[음성 분석 결과]');
      }, 1500);
    }
  };
  
  const toggleAutoVoice = () => {
    setAutoVoice(!autoVoice);
  };

  const toggleVoiceMode = () => {
    setIsVoiceMode(prev => !prev);
    setCommandInput('');
    setIsRecording(false);
  };

  const handleSendCommand = () => {
    if (!commandInput.trim()) return;
    
    console.log(`명령 전송: ${commandInput}`);
    
    // 드론 명령 실행
    if (onDroneCommand) {
      onDroneCommand(commandInput);
    }
  
    // 음성 모드가 아니거나 녹음 중이 아닐 때만 입력 클리어
    if (!isVoiceMode || !isRecording) {
      setCommandInput('');
    }
  };

  const handleCommandInputChange = (value) => {
    setCommandInput(value);
  };

  return {
    isVoiceMode,
    commandInput,
    isRecording,
    autoVoice,
    toggleRecording,
    toggleAutoVoice,
    toggleVoiceMode,
    handleSendCommand,
    handleCommandInputChange,
    setCommandInput
  };
}; 