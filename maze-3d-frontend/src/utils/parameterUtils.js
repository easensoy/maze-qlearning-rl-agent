// Sends only the parameter that changed, so moving one slider (e.g. speed) does not
// overwrite the server's other values (e.g. the decayed epsilon).
export const createParameterHandler = (params, setParams, onUpdate) => {
  return (param, value) => {
    const parsed = parseFloat(value);
    setParams({ ...params, [param]: parsed });
    onUpdate({ [param]: parsed });
  };
};

export const createMazeSizeHandler = (setCurrentMazeSize, onUpdate) => {
  return (width, height, depth, complexityParams = {}) => {
    const newSize = { width, height, depth };
    setCurrentMazeSize(newSize);
    onUpdate({
      maze_width: width,
      maze_height: height,
      maze_depth: depth,
      ...complexityParams
    });
  };
};

// Regenerates the maze at the currently selected size with the new complexity values.
export const createComplexityHandler = (mazeParams, setMazeParams, onUpdate, currentMazeSize) => {
  return (param, value) => {
    const newMazeParams = { ...mazeParams, [param]: parseFloat(value) };
    setMazeParams(newMazeParams);
    onUpdate({
      regenerate: true,
      maze_width: currentMazeSize.width,
      maze_height: currentMazeSize.height,
      maze_depth: currentMazeSize.depth,
      ...newMazeParams
    });
  };
};

export const validateParameterRange = (value, min, max) => {
  return Math.max(min, Math.min(max, parseFloat(value)));
};

export const formatParameterValue = (value, decimals = 2) => {
  return parseFloat(value).toFixed(decimals);
};
