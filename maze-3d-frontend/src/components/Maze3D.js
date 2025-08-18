import React, { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere } from '@react-three/drei';
import * as THREE from 'three';

function MazeWall({ position }) {
  return (
    <Box
      position={[position[0], 1, position[1]]}
      scale={[1, 2, 1]}
      castShadow
      receiveShadow
    >
      <meshStandardMaterial 
        color="#1565C0"
        roughness={0.7}
        metalness={0.0}
      />
    </Box>
  );
}

function Agent({ position, size = 0.4 }) {
  const meshRef = useRef();
  
  return (
    <Sphere
      ref={meshRef}
      position={[position[0], size, position[1]]}
      scale={[size, size, size]}
      castShadow
    >
      <meshStandardMaterial color="#FF0000" />
    </Sphere>
  );
}

function Goal({ position, size = 0.4 }) {
  return (
    <Sphere
      position={[position[0], size, position[1]]}
      scale={[size, size, size]}
      castShadow
    >
      <meshStandardMaterial color="#00FF00" />
    </Sphere>
  );
}

function QValueHeatmap({ qValues, dimensions }) {
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
}

function Maze3D({ mazeData, agentPosition, goalPosition, qValues, showQValues = false }) {
  const cameraRef = useRef();
  
  if (!mazeData || !mazeData.maze) {
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
  
  const { maze, dimensions } = mazeData;
  const { width, height, depth } = dimensions;
  
  const mazeSize = Math.max(width, height);
  const cameraDistance = mazeSize * 0.75;
  const cameraHeight = mazeSize * 0.8;
  const fov = 60;
  
  const walls = [];
  for (let z = 0; z < depth; z++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (maze[z] && maze[z][y] && maze[z][y][x] === 1) {
          walls.push([x, y, z]);
        }
      }
    }
  }
  
  return (
    <div className="maze-viewport">
      <div className="maze-3d-container">
        <Canvas
        shadows
        camera={{
          position: [width/2, cameraHeight, height/2 + cameraDistance/2],  // Perfect centering for auto-fit
          fov: fov,
          near: 0.1,
          far: 1000
        }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[width/2, height + 5, depth/2]}
          intensity={0.8}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-far={50}
          shadow-camera-left={-20}
          shadow-camera-right={20}
          shadow-camera-top={20}
          shadow-camera-bottom={-20}
        />
        
        <mesh position={[(width-1)/2, -0.1, (height-1)/2]} receiveShadow rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[width, height]} />
          <meshLambertMaterial color="#1A1A1A" roughness={0.9} transparent opacity={0.2} />
        </mesh>
        
        {walls.map(([x, y, z], index) => (
          <MazeWall 
            key={`wall-${index}`} 
            position={[x, y]}
          />
        ))}
        
        
        
        {agentPosition && (
          <Agent position={agentPosition} />
        )}
        
        {goalPosition && (
          <Goal position={goalPosition} />
        )}
        
        {showQValues && (
          <QValueHeatmap qValues={qValues} dimensions={dimensions} />
        )}
        
        <OrbitControls
          ref={cameraRef}
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          target={[width/2, 0, height/2]}
          minDistance={cameraDistance * 0.5}
          maxDistance={cameraDistance * 1.8}
          maxPolarAngle={Math.PI * 0.75}
          minPolarAngle={Math.PI * 0.05}
          dampingFactor={0.1}
          enableDamping={true}
        />
        </Canvas>
      </div>
    </div>
  );
}

export default Maze3D;