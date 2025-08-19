import React from 'react';
import { Box } from '@react-three/drei';

const MazeWall = ({ position, color = "#1565C0", scale = [1, 2, 1] }) => {
  return (
    <Box
      position={[position[0], 1, position[1]]}
      scale={scale}
      castShadow
      receiveShadow
    >
      <meshStandardMaterial 
        color={color}
        roughness={0.7}
        metalness={0.0}
      />
    </Box>
  );
};

export default MazeWall;