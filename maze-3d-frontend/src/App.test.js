import { render, screen } from '@testing-library/react';
import App from './App';

// No real WebSocket in unit tests: the hook reports "not connected yet".
jest.mock('./hooks/useWebSocket', () => () => ({
  isConnected: false,
  connectionError: null,
  mazeData: null,
  agentPosition: null,
  goalPosition: null,
  qValues: null,
  trainingStats: null,
  isTraining: false,
  startTraining: jest.fn(),
  stopTraining: jest.fn(),
  resetMaze: jest.fn(),
  updateParameters: jest.fn(),
  initializeMaze: jest.fn()
}));

test('renders the training dashboard while connecting', () => {
  render(<App />);
  expect(screen.getByText(/3D Maze RL Training/i)).toBeInTheDocument();
  expect(screen.getByText(/Start Training/i)).toBeInTheDocument();
  expect(screen.getByText(/Connecting to maze server/i)).toBeInTheDocument();
});
