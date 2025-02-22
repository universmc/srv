const { contextBridge, ipcRenderer } = require('electron');

// Expose une API sécurisée au contexte de la page web
contextBridge.exposeInMainWorld('electronAPI', {
  generateChatCompletion: async (prompt) => {
    try {
      const response = await fetch('http://localhost:3141/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      const data = await response.json();
      return data.message || 'Erreur lors de la génération de la réponse.';
    } catch (error) {
      console.error('Erreur lors de la requête de complétion de chat:', error);
      return 'Erreur de communication avec le serveur.';
    }
  }
});
