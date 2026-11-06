import React, { useMemo } from 'react';
import { Box } from '@react-three/drei';
import * as THREE from 'three';
import { LEVEL_GAP } from '../../utils/constants';

// One floor tile per visited cell, coloured by the best Q-value there:
// red = lowest value in the table, green = highest. Keys from the server are "x,y,z".
const QValueHeatmap = ({ qValues, dimensions }) => {
  const tiles = useMemo(() => {
    const entries = Object.entries(qValues?.best_actions || {});
    if (!dimensions || entries.length === 0) return [];
    const values = entries.map(([, data]) => data.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    return entries.map(([key, data]) => {
      const [x, y, z] = key.split(',').map(Number);
      const t = (data.value - min) / range;
      return {
        key,
        position: [z * (dimensions.width + LEVEL_GAP) + x, 0.06, y],
        color: new THREE.Color().setHSL(0.33 * t, 0.9, 0.5)
      };
    });
  }, [qValues, dimensions]);

  return (
    <group>
      {tiles.map(({ key, position, color }) => (
        <Box key={key} position={position} scale={[0.9, 0.06, 0.9]}>
          <meshBasicMaterial color={color} transparent opacity={0.75} />
        </Box>
      ))}
    </group>
  );
};

export default QValueHeatmap;
