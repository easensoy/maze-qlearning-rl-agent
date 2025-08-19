import React, { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { MazeWall, Agent, Goal, QValueHeatmap } from './3d';

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