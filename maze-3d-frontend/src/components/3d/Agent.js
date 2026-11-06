import React, { useRef } from 'react';
import { Sphere } from '@react-three/drei';

// Drawn after the walls and without depth testing, so the agent stays visible even when a
// wall stands between it and the camera.
const Agent = ({ position, size = 0.4, color = "#FF0000" }) => {
  const meshRef = useRef();

  return (
    <Sphere
      ref={meshRef}
      position={[position[0], size, position[1]]}
      scale={[size, size, size]}
      renderOrder={10}
      castShadow
    >
      <meshStandardMaterial color={color} depthTest={false} />
    </Sphere>
  );
};

export default Agent;
