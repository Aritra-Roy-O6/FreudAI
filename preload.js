const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
    // Communication with Python
    sendToPython: (endpoint, data) => {
        return fetch(`http://127.0.0.1:8000${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        }).then(res => res.json());
    },
    
    // Communication with Electron Vault
    hasKey: () => ipcRenderer.invoke('has-key'),
    saveKey: (key) => ipcRenderer.invoke('save-key', key),
    getKey: () => ipcRenderer.invoke('get-key')
});