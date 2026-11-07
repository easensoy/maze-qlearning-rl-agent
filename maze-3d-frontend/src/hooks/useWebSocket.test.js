import { StrictMode } from 'react';
import { act, renderHook } from '@testing-library/react';
import useWebSocket from './useWebSocket';

// A stand-in for the browser WebSocket that the tests drive by hand.
class FakeSocket {
  static OPEN = 1;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    this.closed = false;
    FakeSocket.instances.push(this);
  }

  send(data) {
    this.sent.push(JSON.parse(data));
  }

  close() {
    this.closed = true;
    this.readyState = 3;
  }

  // Test helpers
  open() {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }

  drop(reason = 'server went away') {
    this.readyState = 3;
    this.onclose?.({ reason });
  }

  receive(message) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }
}

const realWebSocket = global.WebSocket;

beforeEach(() => {
  jest.useFakeTimers();
  jest.spyOn(console, 'log').mockImplementation(() => {});
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  FakeSocket.instances = [];
  global.WebSocket = FakeSocket;
});

afterEach(() => {
  jest.useRealTimers();
  jest.restoreAllMocks();
  global.WebSocket = realWebSocket;
});

const connectHook = (options) => {
  const view = renderHook(() => useWebSocket('ws://test'), options);
  act(() => {
    jest.runOnlyPendingTimers(); // the connect is deferred by one tick
  });
  return view;
};

test('opens one socket and sends commands through it', () => {
  const { result } = connectHook();
  expect(FakeSocket.instances).toHaveLength(1);
  const socket = FakeSocket.instances[0];

  act(() => socket.open());
  expect(result.current.isConnected).toBe(true);

  act(() => {
    result.current.startTraining();
    result.current.updateParameters({ speed: 100 });
  });
  expect(socket.sent).toEqual([
    { type: 'start_training' },
    { type: 'update_params', params: { speed: 100 } }
  ]);
});

test('React StrictMode double mount still opens a single socket', () => {
  connectHook({ wrapper: StrictMode });
  expect(FakeSocket.instances.filter((socket) => !socket.closed)).toHaveLength(1);
});

test('server messages update the hook state', () => {
  const { result } = connectHook();
  const socket = FakeSocket.instances[0];
  act(() => socket.open());

  act(() => socket.receive({
    type: 'maze_state',
    maze: [[[0]]],
    dimensions: { width: 1, height: 1, depth: 1 },
    agent_position: [0, 0, 0],
    goal_position: [0, 0, 0]
  }));
  expect(result.current.mazeData.dimensions).toEqual({ width: 1, height: 1, depth: 1 });

  act(() => socket.receive({
    type: 'training_update',
    stats: { episode: 4 },
    is_training: true,
    step_data: { position: [1, 2, 0] }
  }));
  expect(result.current.trainingStats.episode).toBe(4);
  expect(result.current.isTraining).toBe(true);
  expect(result.current.agentPosition).toEqual([1, 2, 0]);

  act(() => socket.receive({ type: 'training_stopped' }));
  expect(result.current.isTraining).toBe(false);
});

test('after a dropped connection, commands go to the new socket', () => {
  // Regression: the hook used to keep a closed socket as current, so buttons did nothing.
  const { result } = connectHook();
  const first = FakeSocket.instances[0];
  act(() => first.open());

  act(() => first.drop());
  expect(result.current.isConnected).toBe(false);
  expect(result.current.startTraining()).toBe(false); // nothing open: not sent

  act(() => {
    jest.runOnlyPendingTimers(); // reconnect
  });
  const second = FakeSocket.instances[1];
  act(() => second.open());
  act(() => {
    result.current.stopTraining();
  });

  expect(second.sent).toEqual([{ type: 'stop_training' }]);
  expect(first.sent).toEqual([]);
});

test('messages from a replaced socket are ignored', () => {
  const { result } = connectHook();
  const first = FakeSocket.instances[0];
  act(() => first.open());
  act(() => first.drop());
  act(() => {
    jest.runOnlyPendingTimers();
  });

  act(() => first.receive({ type: 'training_update', stats: { episode: 99 }, is_training: true }));
  expect(result.current.trainingStats).toBeNull();
});

test('unmounting closes the socket', () => {
  const { unmount } = connectHook();
  const socket = FakeSocket.instances[0];
  act(() => socket.open());
  unmount();
  expect(socket.closed).toBe(true);
});
