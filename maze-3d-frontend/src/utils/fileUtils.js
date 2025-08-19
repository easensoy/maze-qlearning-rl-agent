export const downloadCSV = (data, filename) => {
  const blob = new Blob([data], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

const getTimestamp = () => new Date().toISOString().slice(0, 19).replace(/:/g, '-');

export const exportQTableToCSV = (qValues) => {
  if (!qValues?.best_actions) return;
  
  const csvContent = "State,Action,Q-Value\n" + 
    Object.entries(qValues.best_actions)
      .map(([state, data]) => `"${state}","${data.action}",${data.value}`)
      .join('\n');
  
  downloadCSV(csvContent, generateTimestampedFilename('q_table', 'csv'));
};

export const generateTimestampedFilename = (baseName, extension) => 
  `${baseName}_${getTimestamp()}.${extension}`;