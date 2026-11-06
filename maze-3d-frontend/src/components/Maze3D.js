import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box } from '@react-three/drei';
import { MazeWall, Agent, Goal, QValueHeatmap } from './3d';
import { LEVEL_GAP } from '../utils/constants';

// Maze cell (x, y, z) -> scene: x runs along X, y along Z, and each level z is placed
// beside the previous one (offset along X), so every level of a multi-level maze stays visible.
const levelOffset = (z, width) => z * (width + LEVEL_GAP);

function Maze3D({ mazeData, agentPosition, goalPosition, qValues, showQValues = false }) {
  const maze = mazeData?.maze;
  const dimensions = mazeData?.dimensions;

  const { walls, shafts } = useMemo(() => {
    if (!maze || !dimensions) return { walls: [], shafts: [] };
    const { width, height, depth } = dimensions;
    const wallElements = [];
    const shaftElements = [];
    for (let z = 0; z < depth; z++) {
      const offset = levelOffset(z, width);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const cell = maze[z]?.[y]?.[x];
          if (cell === 1) {
            wallElements.push(<MazeWall key={`wall-${x}-${y}-${z}`} position={[offset + x, y]} />);
          } else if (cell === 0) {
            const up = z + 1 < depth && maze[z + 1][y][x] === 0;
            const down = z > 0 && maze[z - 1][y][x] === 0;
            if (up || down) {
              shaftElements.push(
                <Box key={`shaft-${x}-${y}-${z}`} position={[offset + x, 0.02, y]} scale={[0.7, 0.04, 0.7]}>
                  <meshStandardMaterial color={up ? '#FFCA28' : '#26C6DA'} />
                </Box>
              );
            }
          }
        }
      }
    }
    return { walls: wallElements, shafts: shaftElements };
  }, [maze, dimensions]);

  if (!maze || !dimensions) {
    return (
      <div className="maze-viewport">
        <div className="maze-3d-container" style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center'
        }}>
          <p style={{ color: '#666', fontSize: '1.2em' }}>Loading 3D maze...</p>
        </div>
      </div>
    );
  }

  const { width, height, depth } = dimensions;
  const totalWidth = depth * width + (depth - 1) * LEVEL_GAP;
  // The viewport is landscape (about 1.4:1), so a wide multi-level layout needs less distance than its width.
  const span = Math.max(totalWidth / 1.4, height);
  const centerX = (totalWidth - 1) / 2;
  const centerZ = (height - 1) / 2;
  const cameraHeight = span * 1.05;
  const cameraDistance = span * 0.5;
  const toScene = (position) => position && [levelOffset(position[2] || 0, width) + position[0], position[1]];

  return (
    <div className="maze-viewport">
      <div className="maze-3d-container">
        <Canvas
          key={`${width}x${height}x${depth}`}
          shadows
          camera={{
            position: [centerX, cameraHeight, centerZ + cameraDistance],
            fov: 60,
            near: 0.1,
            far: 1000
          }}
        >
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[centerX, span + 5, centerZ]}
            intensity={0.8}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
            shadow-camera-far={span * 3}
            shadow-camera-left={-span}
            shadow-camera-right={span}
            shadow-camera-top={span}
            shadow-camera-bottom={-span}
          />

          {Array.from({ length: depth }, (_, z) => (
            <mesh key={`ground-${z}`} position={[levelOffset(z, width) + (width - 1) / 2, -0.1, centerZ]}
                  receiveShadow rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[width, height]} />
              <meshLambertMaterial color="#1A1A1A" transparent opacity={0.2} />
            </mesh>
          ))}

          {walls}
          {shafts}

          {agentPosition && (
            <Agent position={toScene(agentPosition)} />
          )}

          {goalPosition && (
            <Goal position={toScene(goalPosition)} />
          )}

          {showQValues && (
            <QValueHeatmap qValues={qValues} dimensions={dimensions} />
          )}

          <OrbitControls
            enablePan={false}
            enableZoom={true}
            enableRotate={true}
            target={[centerX, 0, centerZ]}
            minDistance={span * 0.4}
            maxDistance={span * 2}
            maxPolarAngle={Math.PI * 0.45}
            minPolarAngle={0}
            dampingFactor={0.1}
            enableDamping={true}
          />
        </Canvas>
      </div>
    </div>
  );
}

export default Maze3D;
