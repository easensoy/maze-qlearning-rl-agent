import React, { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere } from '@react-three/drei';
import * as THREE from 'three';

// Individual 3D wall cube - proper maze geometry
function MazeWall({ position }) {
  return (
    <Box
      position={[position[0], 1, position[1]]}  // Wall height 2 units (positioned at y=1)
      scale={[1, 2, 1]}  // Wall: 1x2x1 (width x height x depth)
      castShadow
      receiveShadow
    >
      <meshStandardMaterial 
        color="#1565C0"  // Blue walls as requested
        roughness={0.7}
        metalness={0.0}
      />
    </Box>
  );
}

// Red agent sphere for maze navigation
function Agent({ position, size = 0.4 }) {
  const meshRef = useRef();
  
  return (
    <Sphere
      ref={meshRef}
      position={[position[0], size, position[1]]}  // Agent at ground level between walls
      scale={[size, size, size]}
      castShadow
    >
      <meshStandardMaterial color="#FF0000" />  {/* Bright red agent */}
    </Sphere>
  );
}

// Green goal cube/sphere for maze target
function Goal({ position, size = 0.4 }) {
  return (
    <Sphere
      position={[position[0], size, position[1]]}  // Goal at ground level
      scale={[size, size, size]}
      castShadow
    >
      <meshStandardMaterial color="#00FF00" />  {/* Bright green goal */}
    </Sphere>
  );
}

// Q-values heatmap overlay - floating above ground in corridors
function QValueHeatmap({ qValues, dimensions }) {
  if (!qValues || !dimensions) return null;
  
  return (
    <group>
      {Object.entries(qValues.best_actions || {}).map(([pos, data]) => {
        const [x, y, z] = pos.split(',').map(Number);
        const intensity = Math.max(0, Math.min(1, data.value / 10)); // Normalize Q-value
        
        return (
          <Box
            key={pos}
            position={[x, y + 0.05, z]} // Float just above ground
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

// Main 3D Maze component
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
  
  // Generate walls only where needed - this creates navigable corridors
  const walls = [];
  for (let z = 0; z < depth; z++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // Only render walls (1), leave corridors (0) as empty space
        if (maze[z] && maze[z][y] && maze[z][y][x] === 1) {
          walls.push([x, y, z]);
        }
      }
    }
  }
  
  // Also identify corridor positions for debugging
  const corridors = [];
  for (let z = 0; z < depth; z++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (maze[z] && maze[z][y] && maze[z][y][x] === 0) {
          corridors.push([x, y, z]);
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
          position: [width/2, Math.max(width, height) + 5, width/2],  // Isometric view from above
          fov: 45,
          near: 0.1,
          far: 1000
        }}
      >
        {/* Lighting */}
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
        
        {/* Minimal ground plane - only for shadows */}
        <mesh position={[width/2, -0.1, height/2]} receiveShadow rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[width * 1.1, height * 1.1]} />
          <meshLambertMaterial color="#1A1A1A" roughness={0.9} transparent opacity={0.3} />
        </mesh>
        
        {/* Proper maze walls - consistent blocks creating clear corridors */}
        {walls.map(([x, y, z], index) => (
          <MazeWall 
            key={`wall-${index}`} 
            position={[x, y]}  // Only pass x,y since walls are at ground level
          />
        ))}
        
        {/* No corridor markers - paths are empty space for clear navigation */}
        
        {/* Agent */}
        {agentPosition && (
          <Agent position={agentPosition} />
        )}
        
        {/* Goal */}
        {goalPosition && (
          <Goal position={goalPosition} />
        )}
        
        {/* Q-values heatmap */}
        {showQValues && (
          <QValueHeatmap qValues={qValues} dimensions={dimensions} />
        )}
        
        {/* Camera controls - better fit-to-viewport */}
        <OrbitControls
          ref={cameraRef}
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          target={[width/2, 0, height/2]}
          minDistance={5}
          maxDistance={Math.max(width, height) * 2}
          maxPolarAngle={Math.PI * 0.8}  // Limit rotation to keep top-down view
        />
        </Canvas>
      </div>
    </div>
  );
}

export default Maze3D;