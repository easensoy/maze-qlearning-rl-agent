import { useState, useEffect, useCallback, useRef } from 'react';
import { WEBSOCKET_CONFIG } from '../utils/constants';

// One WebSocket per mounted app. The socket lives in a ref, so sendMessage is stable and
// always talks to the socket that is actually open. Handlers ignore events from sockets
// that are no longer current (e.g. the one React StrictMode closes during its dev double-mount).
const useWebSocket = (url) => {
  const [isConnected, setIsConnected] = useState(false);
  const [mazeData, setMazeData] = useState(null);
  const [agentPosition, setAgentPosition] = useState(null);
  const [goalPosition, setGoalPosition] = useState(null);
  const [qValues, setQValues] = useState(null);
  const [trainingStats, setTrainingStats] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [connectionError, setConnectionError] = useState(null);

  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);

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

  useEffect(() => {
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        setConnectionError('Failed to create WebSocket connection');
        return;
      }
      socketRef.current = ws;

      ws.onopen = () => {
        if (socketRef.current !== ws) return;
        console.log('WebSocket connected');
        reconnectAttemptsRef.current = 0;
        setIsConnected(true);
        setConnectionError(null);
      };

      ws.onmessage = (event) => {
        if (socketRef.current !== ws) return;
        try {
          handleMessage(JSON.parse(event.data));
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = () => {
        if (socketRef.current !== ws) return;
        setConnectionError('WebSocket connection error');
      };

      ws.onclose = (event) => {
        if (socketRef.current !== ws) return;
        socketRef.current = null;
        setIsConnected(false);
        if (disposed) return;
        console.log('WebSocket disconnected:', event.reason);

        if (reconnectAttemptsRef.current < WEBSOCKET_CONFIG.maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(`Attempting to reconnect... (${reconnectAttemptsRef.current}/${WEBSOCKET_CONFIG.maxReconnectAttempts})`);
          reconnectTimeoutRef.current = setTimeout(
            connect,
            WEBSOCKET_CONFIG.reconnectDelayMultiplier * reconnectAttemptsRef.current
          );
        } else {
          setConnectionError('Failed to connect after multiple attempts');
        }
      };
    };

    // Deferred so React StrictMode's immediate mount/unmount/mount in development
    // does not open (and then abort) an extra socket.
    reconnectTimeoutRef.current = setTimeout(connect, 0);

    return () => {
      disposed = true;
      clearTimeout(reconnectTimeoutRef.current);
      const ws = socketRef.current;
      socketRef.current = null;
      if (ws) {
        ws.close(1000, 'Component unmounted');
      }
    };
  }, [url, handleMessage]);

  const sendMessage = useCallback((message) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      return true;
    }
    console.warn('WebSocket not connected, cannot send message:', message);
    return false;
  }, []);

  const startTraining = useCallback(() => sendMessage({ type: 'start_training' }), [sendMessage]);
  const stopTraining = useCallback(() => sendMessage({ type: 'stop_training' }), [sendMessage]);
  const resetMaze = useCallback(() => sendMessage({ type: 'reset_maze' }), [sendMessage]);

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
