import { useState, useEffect, useCallback, useRef } from 'react';

const useWebSocket = (url) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [mazeData, setMazeData] = useState(null);
  const [agentPosition, setAgentPosition] = useState(null);
  const [goalPosition, setGoalPosition] = useState(null);
  const [qValues, setQValues] = useState(null);
  const [trainingStats, setTrainingStats] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  
  const reconnectTimeoutRef = useRef(null);
  const maxReconnectAttempts = 5;
  const reconnectAttemptsRef = useRef(0);
  
  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionError(null);
        reconnectAttemptsRef.current = 0;
        
        // Initialize maze on connection
        sendMessage({ type: 'init_maze', width: 21, height: 21, depth: 1 });
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.reason);
        setIsConnected(false);
        setSocket(null);
        
        // Attempt to reconnect if not intentional disconnect
        if (!event.wasClean && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(`Attempting to reconnect... (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 2000 * reconnectAttemptsRef.current); // Exponential backoff
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setConnectionError('Failed to connect after multiple attempts');
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionError('WebSocket connection error');
      };
      
      setSocket(ws);
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setConnectionError('Failed to create WebSocket connection');
    }
  }, [url]);
  
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (socket) {
      socket.close(1000, 'Intentional disconnect');
    }
  }, [socket]);
  
  const sendMessage = useCallback((message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message:', message);
    }
  }, [socket]);
  
  const handleMessage = useCallback((data) => {
    switch (data.type) {
      case 'maze_state':
        setMazeData({
          maze: data.maze,
          dimensions: data.dimensions
        });
        setAgentPosition(data.agent_position);
        setGoalPosition(data.goal_position);
        break;
        
      case 'training_update':
        setTrainingStats(data.stats);
        setIsTraining(data.is_training);
        
        if (data.step_data) {
          // Update agent position from step data
          setAgentPosition(data.step_data.position);
        }
        break;
        
      case 'q_values':
        setQValues(data);
        break;
        
      case 'training_started':
        setIsTraining(true);
        break;
        
      case 'training_stopped':
        setIsTraining(false);
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  }, []);
  
  // Connection management
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);
  
  // Training controls
  const startTraining = useCallback(() => {
    sendMessage({ type: 'start_training' });
  }, [sendMessage]);
  
  const stopTraining = useCallback(() => {
    sendMessage({ type: 'stop_training' });
  }, [sendMessage]);
  
  const resetMaze = useCallback(() => {
    sendMessage({ type: 'reset_maze' });
  }, [sendMessage]);
  
  const updateParameters = useCallback((params) => {
    sendMessage({ type: 'update_params', params });
  }, [sendMessage]);
  
  const initializeMaze = useCallback((width, height, depth, complexityParams = {}) => {
    sendMessage({ 
      type: 'init_maze', 
      width, 
      height, 
      depth,
      ...complexityParams
    });
  }, [sendMessage]);
  
  return {
    isConnected,
    connectionError,
    mazeData,
    agentPosition,
    goalPosition,
    qValues,
    trainingStats,
    isTraining,
    startTraining,
    stopTraining,
    resetMaze,
    updateParameters,
    initializeMaze,
    sendMessage
  };
};

export default useWebSocket;