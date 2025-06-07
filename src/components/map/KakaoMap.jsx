import { useEffect, useState } from 'react';
import { Map, MapMarker, ZoomControl } from 'react-kakao-maps-sdk';

const KakaoMap = ({ lat, lng }) => {
  const [currentLocation, setCurrentLocation] = useState({
    lat: lat || 37.5665, // props가 있으면 사용, 없으면 서울 기본값
    lng: lng || 126.9780
  });
  const [locationLoading, setLocationLoading] = useState(true);

  // 현재 위치 가져오기
  useEffect(() => {    
    // props로 위치가 전달되면 현재 위치 가져오기를 건너뛰기
    if (lat && lng) {
      console.log('Props로 위치 전달됨:', lat, lng);
      setLocationLoading(false);
      return;
    }

    console.log('현재 위치 가져오기 시작');
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          console.log('위치 정보 가져오기 성공:', position.coords);
          setCurrentLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
          setLocationLoading(false);
        },
        (error) => {
          console.error('위치 정보를 가져올 수 없습니다:', error);
          setLocationLoading(false); // 기본값 사용
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 600000 // 10분
        }
      );
    } else {
      console.error('이 브라우저는 Geolocation을 지원하지 않습니다.');
      setLocationLoading(false);
    }
  }, [lat, lng]);

  if (locationLoading) {
    return (
      <div style={{ width: '100%', height: '400px' }} className="flex items-center justify-center bg-gray-100 rounded-lg">
        <div className="text-gray-600">위치 정보를 가져오는 중...</div>
      </div>
    );
  }

  return (
    <Map
      center={{ lat: currentLocation.lat, lng: currentLocation.lng }}
      style={{ width: '100%', height: '400px' }}
      level={3}
    >
      <MapMarker position={{ lat: currentLocation.lat, lng: currentLocation.lng }} />
      <ZoomControl position="RIGHT" />
    </Map>
  );
};

export default KakaoMap;