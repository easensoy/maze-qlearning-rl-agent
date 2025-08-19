import React, { useRef } from 'react';
import { Sphere } from '@react-three/drei';

const Agent = ({ position, size = 0.4, color = "#FF0000" }) => {
  const meshRef = useRef();
  
  return (
    <Sphere
      ref={meshRef}
      position={[position[0], size, position[1]]}
      scale={[size, size, size]}
      castShadow
    >
      <meshStandardMaterial color={color} />
    </Sphere>
  );
};

export default Agent;