import React from 'react';
import { Sphere } from '@react-three/drei';

const Goal = ({ position, size = 0.4, color = "#00FF00" }) => {
  return (
    <Sphere
      position={[position[0], size, position[1]]}
      scale={[size, size, size]}
      castShadow
    >
      <meshStandardMaterial color={color} />
    </Sphere>
  );
};

export default Goal;