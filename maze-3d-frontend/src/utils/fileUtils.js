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

export const exportQTableToCSV = (qValues) => {
  if (!qValues || !qValues.best_actions) {
    console.log('No Q-table data available for export');
    return;
  }
  
  let csvContent = "State,Action,Q-Value\n";
  
  Object.entries(qValues.best_actions).forEach(([state, data]) => {
    csvContent += `"${state}","${data.action}",${data.value}\n`;
  });
  
  const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
  const filename = `q_table_${timestamp}.csv`;
  
  downloadCSV(csvContent, filename);
  console.log('Q-table exported to CSV successfully');
};

export const generateTimestampedFilename = (baseName, extension) => {
  const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
  return `${baseName}_${timestamp}.${extension}`;
};