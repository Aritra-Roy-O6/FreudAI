const { app, BrowserWindow, ipcMain, safeStorage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;

// Secure path in the user's OS
const configPath = path.join(app.getPath('userData'), 'freud_config.json');

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false, // CRITICAL: Build it instantly, but keep it invisible
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Safely join the path
  mainWindow.loadFile(path.join(__dirname, 'ui', 'index.html'));
}

function startPython() {
  const enginePath = path.join(__dirname, 'engine');
  const env = { ...process.env };
  
  env.PYTHONPATH = enginePath;
  env.PYTHONIOENCODING = 'utf-8';

  pythonProcess = spawn('python', ['api.py'], {
    cwd: enginePath,
    env: env
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python: ${data.toString()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`Python Log: ${output}`);

    // UX SEQUENCE: Unhide the pre-built window
    if (output.includes("Uvicorn running on")) {
      console.log("Backend is ready. Revealing UI...");
      if (mainWindow) {
        mainWindow.show();
        mainWindow.focus(); // Force Windows to pull it to the front
      }
    }
  });
}

// --- SECURE KEY VAULT (IPC HANDLERS) ---
ipcMain.handle('has-key', () => {
    return fs.existsSync(configPath);
});

ipcMain.handle('save-key', (event, rawKey) => {
    try {
        if (safeStorage.isEncryptionAvailable()) {
            const encryptedBuffer = safeStorage.encryptString(rawKey);
            fs.writeFileSync(configPath, JSON.stringify({ key: encryptedBuffer.toString('base64') }));
            return { success: true };
        } else {
            fs.writeFileSync(configPath, JSON.stringify({ key: rawKey, unencrypted: true }));
            return { success: true };
        }
    } catch (error) {
        console.error("Failed to save key:", error);
        return { success: false, error: error.message };
    }
});

ipcMain.handle('get-key', () => {
    if (!fs.existsSync(configPath)) return null;
    try {
        const data = JSON.parse(fs.readFileSync(configPath));
        if (data.unencrypted) return data.key;
        if (safeStorage.isEncryptionAvailable()) {
            return safeStorage.decryptString(Buffer.from(data.key, 'base64'));
        }
        return null;
    } catch (error) {
        console.error("Failed to decrypt key:", error);
        return null;
    }
});

// INITIALIZATION SEQUENCE
app.whenReady().then(() => {
  createWindow(); // 1. Build the invisible window first
  startPython();  // 2. Start the backend and wait
});

app.on('window-all-closed', () => {
  if (pythonProcess) pythonProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});