export const createParameterHandler = (params, setParams, onUpdate) => {
  return (param, value) => {
    const newParams = { ...params, [param]: parseFloat(value) };
    setParams(newParams);
    onUpdate(newParams);
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

export const createComplexityHandler = (mazeParams, setMazeParams, onUpdate) => {
  return (param, value) => {
    const newMazeParams = { ...mazeParams, [param]: parseFloat(value) };
    setMazeParams(newMazeParams);
    onUpdate({
      regenerate: true,
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