import React from 'react';
import { Box } from '@react-three/drei';
import * as THREE from 'three';

const QValueHeatmap = ({ qValues, dimensions }) => {
  if (!qValues || !dimensions) return null;
  
  return (
    <group>
      {Object.entries(qValues.best_actions || {}).map(([pos, data]) => {
        const [x, y, z] = pos.split(',').map(Number);
        const intensity = Math.max(0, Math.min(1, data.value / 10));
        
        return (
          <Box
            key={pos}
            position={[x, y + 0.05, z]}
            scale={[0.8, 0.05, 0.8]}
            transparent
            opacity={intensity * 0.6}
          >
            <meshBasicMaterial color={new THREE.Color().setHSL(0.3 * intensity, 1, 0.5)} />
          </Box>
        );
      })}
    </group>
  );
};

export default QValueHeatmap;