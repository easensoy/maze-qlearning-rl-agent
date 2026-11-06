import React from 'react';
import { Sphere } from '@react-three/drei';

// Like the agent, drawn on top of the walls so it is always visible.
const Goal = ({ position, size = 0.4, color = "#00FF00" }) => {
  return (
    <Sphere
      position={[position[0], size, position[1]]}
      scale={[size, size, size]}
      renderOrder={9}
      castShadow
    >
      <meshStandardMaterial color={color} depthTest={false} />
    </Sphere>
  );
};

export default Goal;
